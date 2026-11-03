import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  BrandHeader,
  CalibrationPanel,
  CameraPanel,
  CameraPreviews,
  EventLog,
  ExtensionDropdowns,
  RecordingPanel,
  SceneControlsPanel,
  SceneMonitor,
} from "./components/index.js";

async function post(url, body = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `${response.status} ${response.statusText}`);
  return data;
}

export default function Viewer({ navigation }) {
  const [cameras, setCameras] = useState([]);
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState("Подключаемся к серверу…");
  const [sceneNotice, setSceneNotice] = useState("Ожидание соединения с сервером");
  const [mode, setMode] = useState("—");
  const [detector, setDetector] = useState("—");
  const [telemetry, setTelemetry] = useState({});
  const [people, setPeople] = useState([]);
  const [logs, setLogs] = useState([]);
  const [measure, setMeasure] = useState("");
  const [showMeasure, setShowMeasure] = useState(false);
  const [showBoard, setShowBoard] = useState(false);
  const [showTextures, setShowTextures] = useState(true);
  const [showSkeletons, setShowSkeletons] = useState(true);
  const [recording, setRecording] = useState(false);
  const [jobs, setJobs] = useState({});
  const lastFrameStatusAt = useRef(0);

  const addLog = useCallback((message) => {
    setLogs((current) => [`${new Date().toLocaleTimeString()}  ${message}`, ...current].slice(0, 80));
  }, []);

  useEffect(() => {
    const engineUrl = new URL("/static/viewer-engine.js", location.origin).href;
    import(/* @vite-ignore */ engineUrl).catch((error) => {
      setConnected(false);
      setStatus("Не удалось загрузить 3D-сцену");
      setSceneNotice("Не удалось загрузить 3D-сцену");
      addLog(`3D-сцена: ${error.message}`);
    });
  }, [addLog]);

  useEffect(() => {
    const onMessage = (event) => {
      const message = event.detail;
      if (message.type === "hello") {
        setMode(message.mode || "—");
        setDetector(message.detector || "—");
        setRecording(Boolean(message.recording));
        setCameras(message.cameras || []);
        setConnected(true);
        setStatus("Сервер подключён");
        setSceneNotice("");
      } else if (message.type === "cameras") {
        setCameras(message.cameras || []);
      } else if (message.type === "frame") {
        const now = Date.now();
        if (now - lastFrameStatusAt.current >= 1000) {
          lastFrameStatusAt.current = now;
          const stats = message.stats || {};
          setConnected(true);
          setStatus(`LIVE · ${stats.fps ?? "—"} к/с`);
          setSceneNotice("");
          setTelemetry({
            fps: stats.fps,
            latency: stats.latency_ms,
            skew: stats.skew_ms,
            detect: stats.detect_ms,
            usedCameras: stats.cams_used || [],
          });
          setPeople(message.persons || []);
        }
      }
    };
    const onCameras = (event) => setCameras(event.detail || []);
    const onStatus = (event) => {
      const online = !String(event.detail).toLowerCase().includes("нет связи");
      setConnected(online);
      if (!online) {
        setSceneNotice("Связь потеряна · переподключение…");
        setTelemetry({});
        setPeople([]);
      }
      setStatus(event.detail);
    };
    const onLog = (event) => addLog(event.detail);
    const onMeasure = (event) => {
      setShowMeasure(event.detail.visible);
      setMeasure(event.detail.text);
    };
    window.addEventListener("peleng:message", onMessage);
    window.addEventListener("peleng:cameras", onCameras);
    window.addEventListener("peleng:status", onStatus);
    window.addEventListener("peleng:log", onLog);
    window.addEventListener("peleng:measure", onMeasure);
    return () => {
      window.removeEventListener("peleng:message", onMessage);
      window.removeEventListener("peleng:cameras", onCameras);
      window.removeEventListener("peleng:status", onStatus);
      window.removeEventListener("peleng:log", onLog);
      window.removeEventListener("peleng:measure", onMeasure);
    };
  }, [addLog]);

  useEffect(() => {
    window.peleng?.setShowSkeletons(showSkeletons);
  }, [showSkeletons]);
  useEffect(() => {
    window.peleng?.setShowTextures(showTextures);
  }, [showTextures]);

  const run = async (label, action) => {
    try {
      await action();
    } catch (error) {
      addLog(`${label}: ${error.message}`);
    }
  };

  const calibrateExtrinsics = () => run("Калибровка", async () => {
    addLog("Калибровка положения камер по доске…");
    const result = await post("/api/calib/extrinsics");
    for (const [id, camera] of Object.entries(result.cameras || {})) {
      addLog(camera.ok
        ? `${id}: готово · ошибка ${camera.reproj_px} px`
        : `${id}: ${camera.message}`);
    }
    if (result.message) addLog(result.message);
  });

  const scanRoom = () => run("Сканирование", async () => {
    addLog("Сканирую комнату…");
    const result = await post("/api/room/scan");
    addLog(`Облако точек: ${result.points}. ${result.message}`);
  });

  const startIntrinsics = (id) => run(`${id} · оптика`, async () => {
    await post(`/api/calib/${encodeURIComponent(id)}/intrinsics/start`, { n: 15 });
    addLog(`${id}: показывайте калибровочную доску под разными углами`);
    setJobs((current) => ({ ...current, [id]: "Сбор кадров: 0/15" }));
    while (true) {
      await new Promise((resolve) => setTimeout(resolve, 500));
      const response = await fetch(`/api/calib/${encodeURIComponent(id)}/intrinsics`);
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || response.statusText);
      const text = result.state === "collecting"
        ? `Сбор кадров: ${result.n}/${result.need}`
        : result.state === "done"
          ? `Готово · ошибка ${result.rms} px`
          : `${result.state} ${result.message || ""}`;
      setJobs((current) => ({ ...current, [id]: text }));
      if (result.state !== "collecting") break;
    }
  });

  const resetCamera = (id) => run(`${id} · сброс`, async () => {
    await post(`/api/calib/${encodeURIComponent(id)}/reset`, { keep_intrinsics: true });
    addLog(`${id}: положение сброшено`);
  });

  const updateHfov = (id, value) => run(`${id} · обзор`, async () => {
    await post(`/api/calib/${encodeURIComponent(id)}/hfov`, { hfov_deg: value });
    addLog(`${id}: угол обзора ${value}°`);
  });

  const toggleRecording = () => run("Запись", async () => {
    const result = await post(recording ? "/api/record/stop" : "/api/record/start");
    setRecording(!recording);
    addLog(`${recording ? "Сессия сохранена" : "Запись началась"}: ${result.file}`);
  });

  const resetTrails = () => run("Траектории", async () => {
    await post("/api/tracks/reset");
    window.peleng?.clearTrails();
    addLog("Траектории сброшены");
  });

  const clearRoomPoints = () => run("Комната", async () => {
    await post("/api/room/clear");
    window.peleng?.clearRoomPoints();
    addLog("Облако точек очищено");
  });

  return (
    <main className="app-shell">
      <BrandHeader connected={connected} status={status} mode={mode} detector={detector}>{navigation}</BrandHeader>

      <div className="workspace">
        <section className="visual-area">
          <SceneMonitor
            connected={connected}
            telemetry={telemetry}
            people={people}
            sceneNotice={sceneNotice}
            showMeasure={showMeasure}
            measure={measure}
          />
          <CameraPreviews cameras={cameras} showBoard={showBoard} />
        </section>

        <aside className="sidebar">
          <CalibrationPanel
            showBoard={showBoard}
            onShowBoard={setShowBoard}
            onCalibrate={calibrateExtrinsics}
          />
          <CameraPanel
            cameras={cameras}
            jobs={jobs}
            onIntrinsics={startIntrinsics}
            onReset={resetCamera}
            onHfov={updateHfov}
          />
          <SceneControlsPanel
            showTextures={showTextures}
            showSkeletons={showSkeletons}
            onShowTextures={setShowTextures}
            onShowSkeletons={setShowSkeletons}
            onScan={scanRoom}
            onClearRoom={clearRoomPoints}
            onResetTrails={resetTrails}
            onSetMark={() => window.peleng?.setMark()}
          />
          <RecordingPanel recording={recording} onToggle={toggleRecording} />

          <div id="ext" className="extensions" />
          <ExtensionDropdowns />

          <EventLog logs={logs} />
          <footer className="sidebar-footer">PELENG · MULTI-CAMERA SPATIAL SYSTEM</footer>
        </aside>
      </div>
    </main>
  );
}
