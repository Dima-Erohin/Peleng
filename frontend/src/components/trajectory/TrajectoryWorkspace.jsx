import React, { useEffect, useRef, useState } from "react";
import { BrandHeader } from "../BrandHeader";
import { Button } from "../ui";
import { Select } from "../Dropdown";
import PlaybackControls, { PlaybackIcon } from "./PlaybackControls";
import CoordinateChart from "./CoordinateChart";
import TrackingPerformance from "./TrackingPerformance";
import CameraVideo from "./CameraVideo";
import ReconstructionPanel from "./ReconstructionPanel";
import MotionWaveform from "./MotionWaveform";
import Minimap from "./Minimap";
import { createDataset, fetchText, loadDemo, MAX_TRACKS, MAX_TOTAL_SAMPLES, parseTrajectoryCsv, parseTrajectorySvg } from "../../trajectory/data.js";
import { parseImageTrack } from "../../trajectory/image-tracks.js";
import { reconstructDataset } from "../../trajectory/reconstruction.js";
import { buildOnlineTrack, onlineClock, onlineStats, ONLINE_TRACK_ID } from "../../trajectory/online.js";
import { observationAt, trackingQuality } from "../../trajectory/observations.js";

const EMPTY_SAMPLE = { x: 0, y: 0, z: 0, speed: 0, distance: 0, visible: false };

export default function TrajectoryWorkspace({ active, navigation }) {
  const containerRef = useRef(null);
  const engineRef = useRef(null);
  const dataInputRef = useRef(null);
  const videoInputRef = useRef(null);
  const observationInputRef = useRef(null);
  const settingsToggleRef = useRef(null);
  const objectUrls = useRef([]);
  const request = useRef(null);
  const [engineReady, setEngineReady] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [dataset, setDataset] = useState(null);
  const [clips, setClips] = useState([]);
  const [time, setTime] = useState(0);
  const [sample, setSample] = useState(EMPTY_SAMPLE);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const [loop, setLoop] = useState(true);
  const [follow, setFollow] = useState(true);
  const [showRoute, setShowRoute] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [selectedId, setSelectedId] = useState("");
  const [hiddenTracks, setHiddenTracks] = useState([]);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [fps, setFps] = useState(30);
  const [units, setUnits] = useState("cm");
  const [url, setUrl] = useState("");
  // Онлайн-прогон (offline/online.py): есть, если страницу отдаёт онлайн-сервер
  const [online, setOnline] = useState(null);
  const [videoServer, setVideoServer] = useState(false);
  const [liveTrack, setLiveTrack] = useState(null);
  const onlineRef = useRef({ ws: null, hello: null, points: [] });
  const track = selectedId === ONLINE_TRACK_ID ? liveTrack : dataset?.tracks.find((item) => item.id === selectedId);
  const pendingObjects = dataset?.reconstructions?.filter((item) => !item.hasTrack) || [];
  const selectedObservation = dataset?.reconstructions?.find((item) => item.id === selectedId);
  const truckPoint = online ? online.stats?.last : observationAt(dataset?.truckObservations?.points || [], time);
  const quality = trackingQuality(truckPoint, time);

  useEffect(() => {
    if (!settingsOpen) return;
    const onKeyDown = (event) => {
      if (event.key === "Escape") { setSettingsOpen(false); settingsToggleRef.current?.focus(); }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [settingsOpen]);

  const releaseVideos = () => {
    objectUrls.current.forEach((value) => URL.revokeObjectURL(value));
    objectUrls.current = [];
  };

  useEffect(() => {
    let cancelled = false;
    import("../../trajectory/scene.js").then(({ TrajectoryScene }) => {
      if (cancelled) return;
      const engine = new TrajectoryScene(containerRef.current, {
        onFrame: (frame) => { setTime(frame.time); setSample(frame.sample || EMPTY_SAMPLE); },
        onStats: setStats,
        onPlayback: setPlaying,
        onError: setError,
        onModelReady: () => setModelReady(true),
      });
      engineRef.current = engine;
      setEngineReady(true);
    }).catch((reason) => { if (!cancelled) setError(`Не удалось создать 3D-сцену: ${reason.message}`); });
    return () => {
      cancelled = true;
      engineRef.current?.dispose();
      engineRef.current = null;
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    request.current = controller;
    loadDemo(controller.signal).then((value) => {
      if (!controller.signal.aborted) { setDataset(reconstructDataset(value)); setClips(value.clips); setLoading(false); }
    }).catch((reason) => {
      if (!controller.signal.aborted) { setError(`Демо: ${reason.message}`); setLoading(false); }
    });
    return () => { request.current?.abort(); releaseVideos(); };
  }, []);

  useEffect(() => {
    if (!dataset || !engineReady) return;
    const engine = engineRef.current;
    engine.setPlaying(false);
    setModelReady(false);
    setHiddenTracks([]);
    setSelectedId(dataset.tracks[0].id);
    engine.setDataset(dataset);
  }, [dataset, engineReady]);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/online/state", { signal: controller.signal, cache: "no-store" })
      .then((response) => response.ok ? response.json() : null)
      .then((info) => {
        if (info?.type !== "hello") return;
        setVideoServer(true);
        // онлайн-расчёт — только по ссылке /?online=1; без неё — воспроизведение готовых траекторий
        if (new URLSearchParams(location.search).get("online") === "1") setOnline({ info, status: "подключение", stats: null });
      })
      .catch(() => {});
    return () => controller.abort();
  }, []);

  // Страницу отдаёт offline/online.py: видео — с него (в сборке их может не быть). Онлайн-режим — без готовой
  // offline-оценки: её считает сервер сейчас.
  useEffect(() => {
    if (!videoServer || !dataset || dataset.serverClips || dataset.name !== "Project 02") return;
    const serverClips = dataset.clips.map((clip, i) => ({ ...clip, url: `/api/online/video/${i}` }));
    setDataset({ ...dataset, serverClips: true, online: Boolean(online), clips: serverClips,
      tracks: online ? dataset.tracks.filter((item) => item.source !== "estimate") : dataset.tracks });
    setClips(serverClips);
  }, [videoServer, online, dataset]);

  useEffect(() => {
    if (!online || !dataset?.online || !engineReady) return;
    const engine = engineRef.current;
    const state = onlineRef.current;
    const showTrack = () => {
      const value = buildOnlineTrack(state.points);
      if (value) {
        const first = !engine.liveTrack;
        engine.setLiveTrack(value);
        setLiveTrack(value);
        if (first) { engine.select(ONLINE_TRACK_ID); setSelectedId(ONLINE_TRACK_ID); }
      }
      setOnline((current) => current && { ...current, stats: onlineStats(state.points), points: [...state.points] });
    };
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/online`);
    state.ws = ws;
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === "hello") {
        state.hello = message;
        state.points = message.points || [];
        engine.setLiveTrack(null);
        setLiveTrack(null);
        engine.liveClock = message.t0 && !message.done ? onlineClock(message) : null;
        engine.setPlaying(!message.done);
        setOnline((current) => current && { ...current, info: message, status: message.done ? "завершён" : "идёт" });
        showTrack();
        if (message.done && state.points.length) engine.seek(state.points.at(-1).t);
      } else if (message.type === "point" && message.session === state.hello?.session) {
        state.points.push(message);
        showTrack();
      } else if (message.type === "done" && message.session === state.hello?.session) {
        engine.liveClock = null;
        if (state.points.length) engine.seek(state.points.at(-1).t);
        engine.setPlaying(false);
        setOnline((current) => current && { ...current, status: "завершён" });
      }
    };
    ws.onerror = () => setError("Нет соединения с онлайн-сервером. Проверьте запуск сервера и перезагрузите страницу.");
    ws.onclose = () => {
      engine.liveClock = null;
      engine.setPlaying(false);
      setOnline((current) => current && { ...current, status: "нет связи" });
    };
    return () => { ws.onclose = null; ws.onerror = null; ws.close(); engine.liveClock = null; state.ws = null; };
  }, [online !== null, dataset?.online, engineReady]);

  const restartOnline = () => onlineRef.current.ws?.readyState === WebSocket.OPEN && onlineRef.current.ws.send(JSON.stringify({ type: "start" }));

  const toggleRoute = () => {
    const show = !showRoute;
    setShowRoute(show);
    if (show) {
      setFollow(false);
      engineRef.current.follow = false;
      engineRef.current.focus(true);
    }
  };

  useEffect(() => { engineRef.current?.setActive(active); }, [active, engineReady]);
  useEffect(() => { engineRef.current?.setShowRoute(showRoute); }, [showRoute, engineReady]);
  useEffect(() => { if (engineRef.current) engineRef.current.rate = rate; }, [rate, engineReady]);
  useEffect(() => { if (engineRef.current) engineRef.current.loop = loop; }, [loop, engineReady]);
  useEffect(() => {
    if (!engineRef.current) return;
    engineRef.current.follow = follow;
    if (follow && dataset) engineRef.current.focus();
  }, [follow, engineReady, dataset]);

  const seek = (value) => engineRef.current?.seek(value);
  const play = () => engineRef.current?.setPlaying(!engineRef.current.playing);
  const step = (direction) => {
    const engine = engineRef.current;
    if (!engine || !dataset) return;
    engine.setPlaying(false);
    engine.seek(engine.time + direction / dataset.fps);
  };
  const selectTrack = (id) => { setSelectedId(id); engineRef.current?.select(id); };

  const beginLoad = () => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setLoading(true);
    setError("");
    engineRef.current?.setPlaying(false);
    return controller;
  };
  const acceptDataset = (value) => {
    releaseVideos();
    setDataset(value);
    setClips(value.clips);
  };
  const readTracks = (text, name, id) => name.toLowerCase().endsWith(".svg") || text.trimStart().startsWith("<")
    ? parseTrajectorySvg(text, name)
    : [parseTrajectoryCsv(text, { fps, units, id, name })];

  const loadFiles = async (files) => {
    if (!files.length) return;
    const controller = beginLoad();
    try {
      if (files.length > MAX_TRACKS) throw new Error(`Максимум ${MAX_TRACKS} файлов объектов`);
      const tracks = [];
      for (const [index, file] of files.entries()) {
        if (file.size > 32 * 1024 * 1024) throw new Error(`${file.name}: файл больше 32 МБ`);
        tracks.push(...readTracks(await file.text(), file.name, `object-${index + 1}`));
        if (tracks.length > MAX_TRACKS || tracks.reduce((sum, item) => sum + item.times.length, 0) > MAX_TOTAL_SAMPLES) {
          throw new Error(`Максимум ${MAX_TRACKS} объектов и ${MAX_TOTAL_SAMPLES} отсчётов на сцену`);
        }
      }
      if (!controller.signal.aborted) acceptDataset(createDataset(tracks, { name: files.length === 1 ? files[0].name : `${files.length} файлов`, fps }));
    } catch (reason) {
      if (!controller.signal.aborted) setError(`Импорт: ${reason.message}`);
    } finally { if (!controller.signal.aborted) setLoading(false); }
  };

  const loadUrl = async () => {
    const controller = beginLoad();
    try {
      const source = new URL(url, location.href);
      if (!["http:", "https:"].includes(source.protocol)) throw new Error("Укажите HTTP(S)-адрес файла");
      const text = await fetchText(source.href, controller.signal);
      if (!controller.signal.aborted) acceptDataset(createDataset(readTracks(text, source.pathname, "object-1"), { name: "Данные бэкенда", fps }));
    } catch (reason) {
      if (!controller.signal.aborted) setError(`Источник: ${reason.message}`);
    } finally { if (!controller.signal.aborted) setLoading(false); }
  };

  const restoreDemo = async () => {
    const controller = beginLoad();
    try {
      const value = await loadDemo(controller.signal);
      if (!controller.signal.aborted) { acceptDataset(reconstructDataset(value)); setFps(30); setUnits("cm"); }
    } catch (reason) {
      if (!controller.signal.aborted) setError(`Демо: ${reason.message}`);
    } finally { if (!controller.signal.aborted) setLoading(false); }
  };

  const loadVideos = (files) => {
    if (!files.length) return;
    releaseVideos();
    const nextClips = files.map((file, i) => {
      const objectUrl = URL.createObjectURL(file);
      objectUrls.current.push(objectUrl);
      return { id: `video-${i}`, name: file.name, url: objectUrl };
    });
    setClips(nextClips);
  };
  const loadObservations = async (files) => {
    if (!files.length || !dataset) return;
    const controller = beginLoad();
    try {
      if (files.length > MAX_TRACKS) throw new Error(`Максимум ${MAX_TRACKS} файлов наблюдений`);
      const merged = new Map(dataset.imageTracks.map((item) => [JSON.stringify([item.object_id, item.camera_id]), item]));
      for (const file of files) {
        if (file.size > 32 * 1024 * 1024) throw new Error(`${file.name}: файл больше 32 МБ`);
        const item = parseImageTrack(JSON.parse(await file.text()));
        merged.set(JSON.stringify([item.object_id, item.camera_id]), item);
      }
      if (!controller.signal.aborted) setDataset(reconstructDataset(dataset, [...merged.values()]));
    } catch (reason) {
      if (!controller.signal.aborted) setError(`Наблюдения: ${reason.message}`);
    } finally { if (!controller.signal.aborted) setLoading(false); }
  };
  const toggleTrack = (id) => {
    engineRef.current?.setObjectVisible(id, hiddenTracks.includes(id));
    setHiddenTracks((values) => values.includes(id) ? values.filter((value) => value !== id) : [...values, id]);
  };
  const ready = Boolean(dataset && engineReady && modelReady && !loading);
  const closeSettings = () => { setSettingsOpen(false); settingsToggleRef.current?.focus(); };

  return (
    <main className="app-shell replay-shell dashboard-shell">
      <BrandHeader connected={ready} status={loading ? "Загрузка данных…" : online ? `Онлайн-прогон · ${online.status}` : playing ? "Воспроизведение" : "Траектории · пауза"}
        mode={online ? "Онлайн" : "Траектории"} detector={online ? `${online.info.weights} · ${online.info.device}` : "CSV / SVG / 2D → 3D"}>{navigation}</BrandHeader>
      <div className="dashboard-toolbar">
        <div className="dashboard-object-select"><span>ОБЪЕКТ</span><Select label="Выбранный объект" value={selectedId} onChange={selectTrack} disabled={!ready}
          options={[...(dataset?.tracks || []), ...(liveTrack ? [liveTrack] : []), ...pendingObjects].map((item) => ({ value: item.id, label: item.name }))} /></div>
        <span className="dashboard-mode">{online ? `${online.status} · ${online.stats?.rate?.toFixed(1) || "—"} Гц · ${online.stats?.lagMs ?? "—"} мс` : "ЗАПИСЬ · СИНХРОННОЕ ВОСПРОИЗВЕДЕНИЕ"}</span>
        <button type="button" className="dashboard-settings-toggle" ref={settingsToggleRef} aria-expanded={settingsOpen} aria-controls="dashboard-settings"
          onClick={() => setSettingsOpen(!settingsOpen)}>Данные и объекты <span>{settingsOpen ? "×" : "+"}</span></button>
      </div>
      <div className="replay-workspace">
        <section className="replay-main">
          <div className="replay-scene">
            <div className="replay-canvas" ref={containerRef} />
            {error && <div className="replay-error dashboard-error" role="alert"><b>Не удалось выполнить действие</b><span>{error}</span><button type="button" onClick={() => setError("")} aria-label="Закрыть сообщение">×</button></div>}
            <div className="replay-scene-top">
              <div className="scene-title"><span>ПРОСТРАНСТВЕННЫЙ ТРЕКИНГ</span><h1>{dataset?.name || "Загрузка сцены"}</h1>
                <p>{dataset ? `Объектов: ${dataset.tracks.length + pendingObjects.length} · камер: ${dataset.cameras.length} · Z вверх · метры` : "Подготовка траекторий и 3D-модели"}</p></div>
              <span className="replay-data-badge"><i />{dataset?.imageTracks.length ? "CSV + ВИДЕОНАБЛЮДЕНИЯ" : "ТРАЕКТОРИИ XYZ"}</span>
            </div>
            <div className="scene-tool-stack" role="group" aria-label="Управление 3D-камерой">
              <button type="button" disabled={!ready} className={follow ? "is-active" : ""} aria-label="Следовать за объектом" aria-pressed={follow}
                onClick={() => setFollow(!follow)}><PlaybackIcon name="target" /><span>Следовать</span></button>
              <button type="button" disabled={!ready} aria-label="Вся траектория" onClick={() => { setFollow(false); engineRef.current.follow = false; engineRef.current.focus(true); }}><PlaybackIcon name="overview" /><span>Обзор</span></button>
              <button type="button" disabled={!ready} className={showRoute ? "is-active" : ""} aria-label="Показать весь маршрут" aria-pressed={showRoute}
                onClick={toggleRoute}><PlaybackIcon name="route" /><span>Маршрут</span></button>
              <button type="button" disabled={!ready} aria-label="Приблизить объект" onClick={() => engineRef.current?.focus()}><PlaybackIcon name="target" /><span>К объекту</span></button>
            </div>
            {!modelReady && !error && <div className="scene-loading" role="status"><span className="loading-dot" />{loading ? "Читаем траектории" : "Загружаем CargoTruck.fbx"}</div>}
            <div className="replay-scene-bottom">
              <div className="scene-object-tag">
                <i style={{ background: track?.color || "#69b9ed" }} />
                <span>{track?.name || selectedObservation?.name || "Объект"}<small>{sample.visible ? selectedObservation ? "Триангуляция по видео" : track?.source === "estimate" ? "Сглаженная 3D · исходные XYZ в графиках" : track?.live ? "Онлайн-оценка по видео" : "Исходная траектория CSV / SVG" : selectedObservation ? "XYZ не определены" : "Нет отсчёта"}</small></span>
                <b>{sample.visible && sample.motionAvailable !== false ? (sample.speed * 3.6).toFixed(1) : "—"}<small>км/ч</small></b>
                <div className="scene-motion-wave">
                  <div><span>ДВИЖЕНИЕ</span><span>{track ? "ПОСЛЕДНИЕ 4 С" : "НЕТ XYZ"}</span></div>
                  <MotionWaveform track={track} time={time} />
                </div>
              </div>
              <div className="render-stats" aria-label="Производительность рендера"><b>{playing ? stats?.fps ?? "—" : "IDLE"}</b> {playing ? "FPS" : "RENDER"}<span>p95 {stats?.frameMs ?? "—"} мс</span></div>
            </div>
            <div className="scene-axis" aria-label="Оси координат"><span>X</span><span>Y</span><span>Z</span></div>
          </div>
          <PlaybackControls dataset={dataset} time={time} playing={playing} rate={rate} loop={loop} disabled={!ready || Boolean(online)} fps={dataset?.fps || 30}
            onPlay={play} onSeek={seek} onStep={step} onRate={setRate} onLoop={setLoop} />
        </section>
        <section className="dashboard-media" aria-label="Видео и пространственный обзор">
          <CameraVideo clips={clips} imageTracks={dataset?.imageTracks || []} time={time} playing={playing} rate={rate} active={active}
            observations={dataset?.truckObservations} onlinePoint={online?.stats?.last} online={Boolean(online)} onlineInfo={online?.info} />
          {dataset && <Minimap dataset={dataset} liveTrack={liveTrack} hiddenTracks={hiddenTracks}
            selectedId={selectedId} time={time} onSelect={selectTrack} />}
        </section>
        <section className="dashboard-telemetry" aria-label="Телеметрия выбранного объекта">
          <div className="dashboard-position">
            <div className="replay-section-heading"><h2>Положение объекта</h2><span>XYZ · МЕТРЫ</span></div>
            <div className="xyz-readout">
              {["x", "y", "z"].map((axis) => <div key={axis}><span>{axis.toUpperCase()}</span><b>{sample.visible ? Math.abs(sample[axis]) < 0.005 ? "0.00" : sample[axis].toFixed(2) : "—"}</b><small>м</small></div>)}
            </div>
          </div>
          <div className="dashboard-motion">
            <div className="replay-section-heading"><h2>Движение</h2><span>ИСХОДНЫЕ XYZ</span></div>
            <div className="motion-readout"><span>Скорость<b>{sample.visible && sample.motionAvailable !== false ? sample.speed.toFixed(2) : "—"} <small>м/с</small></b></span><span>Пройдено<b>{sample.visible && sample.motionAvailable !== false ? sample.distance.toFixed(1) : "—"} <small>м</small></b></span></div>
            {(online || dataset?.truckObservations?.trajectoryMatches) && <span className={`dashboard-quality is-${quality.level}`}>{quality.label}</span>}
          </div>
          <TrackingPerformance dataset={dataset} points={online?.points} live={Boolean(online)} time={time} />
        </section>
        <aside id="dashboard-settings" className="replay-sidebar dashboard-settings" hidden={!settingsOpen} aria-label="Данные и объекты">
          <div className="dashboard-settings-heading"><h2>Данные и объекты</h2><button type="button" aria-label="Закрыть настройки" onClick={closeSettings}>×</button></div>
          {online && (
            <section className="replay-source online-run" aria-label="Онлайн-прогон">
              <div className="replay-section-heading"><h2>Онлайн-прогон</h2><span>{online.status.toUpperCase()}</span></div>
              <div className="source-card"><span className="source-file-icon online-icon">LIVE</span><div><b>Peleng · онлайн</b>
                <small>{online.stats?.count ? `Точек: ${online.stats.count}${online.stats.rate ? ` · ${online.stats.rate.toFixed(1)} в секунду` : ""}${online.stats.lagMs != null ? ` · задержка ${(online.stats.lagMs / 1000).toFixed(1)} с` : ""}` : "Ждём первую точку"}</small></div><i /></div>
              <p className="online-note">Координаты считает сервер в момент показа видео: детектор позы грузовика ({online.info.weights}, {online.info.device}),
                {online.info.fallback === "roi" ? " поиск в окне вокруг прогноза," : ""} 3D-трекер. Будущих точек нет — линия растёт вслед за грузовиком. Cargo Truck · 01 — эталон для сравнения.</p>
              <div className="source-buttons">
                <Button onClick={restartOnline}>Перезапустить</Button>
                <Button onClick={() => selectTrack(ONLINE_TRACK_ID)} disabled={!liveTrack}>К онлайн-объекту</Button>
              </div>
            </section>
          )}
          <section className="replay-source">
            <div className="replay-section-heading"><h2>Источник данных</h2><span>01</span></div>
            <div className="source-card"><span className="source-file-icon">XYZ</span><div><b>{dataset?.name || "Нет данных"}</b><small>{track ? `Отсчётов: ${track.times.length.toLocaleString("ru")} · ${(dataset.duration).toFixed(2)} с` : selectedObservation ? `Наблюдения камер · XYZ: ${selectedObservation.validCount}` : "CSV или SVG с временными отметками"}</small></div><i /></div>
            {(online || dataset?.truckObservations?.trajectoryMatches) && <div className={`tracking-quality is-${quality.level}`} role="status" aria-label="Надёжность оценки грузовика">
              <b>{quality.label}</b><span>{quality.detail}</span>
              <small>{truckPoint ? `Кадр ${truckPoint.frame} · σ ${Number.isFinite(truckPoint.sigma_m) ? `${truckPoint.sigma_m.toFixed(2)} м` : "—"}` : "Ожидание расчёта"}</small>
              <small>σ — оценка фильтра, не ошибка относительно эталона.</small>
            </div>}
            {track?.skippedFrames?.length > 0 && <p className="source-warning" role="status">В исходном CSV повреждены {track.skippedFrames.length} координаты (кадры {track.skippedFrames.join(", ")}). Строки пропущены; между соседними отсчётами показана интерполяция.</p>}
            <div className="source-buttons">
              <Button onClick={() => dataInputRef.current?.click()} disabled={loading}>Загрузить CSV / SVG</Button>
              <Button onClick={() => videoInputRef.current?.click()}>Видео</Button>
            </div>
            <input ref={dataInputRef} type="file" accept=".csv,.svg" multiple hidden onChange={(event) => { loadFiles([...event.target.files]); event.target.value = ""; }} />
            <input ref={videoInputRef} type="file" accept="video/mp4,video/webm" multiple hidden onChange={(event) => { loadVideos([...event.target.files]); event.target.value = ""; }} />
            <input ref={observationInputRef} type="file" accept=".json" aria-label="Наблюдения камер JSON" multiple hidden onChange={(event) => { loadObservations([...event.target.files]); event.target.value = ""; }} />
            <details className="source-settings">
              <summary>Параметры импорта и URL бэкенда</summary>
              <div className="source-settings-body">
                <label>Частота CSV, кадр/с<input type="number" min="1" max="240" step="any" value={fps} onChange={(event) => setFps(Number(event.target.value))} /></label>
                <div><span className="source-field-label">Единицы CSV</span><Select label="Единицы CSV" value={units} onChange={setUnits}
                  options={[{ value: "cm", label: "Сантиметры" }, { value: "m", label: "Метры" }, { value: "mm", label: "Миллиметры" }]} /></div>
                <p>Параметры применяются при следующем импорте. SVG содержит собственные время и единицы.</p>
                <label>URL траекторий<input type="url" placeholder="/api/trajectories/export.svg" value={url} onChange={(event) => setUrl(event.target.value)} /></label>
                <Button onClick={loadUrl} disabled={loading || !url.trim()}>Получить с бэкенда</Button>
                <Button onClick={restoreDemo} disabled={loading}>Вернуть демо Project 02</Button>
              </div>
            </details>
          </section>
          {dataset?.cameras.length > 0 && <ReconstructionPanel reconstructions={dataset.reconstructions} selectedId={selectedId}
            time={time} onSelect={selectTrack} onImport={() => observationInputRef.current?.click()} disabled={loading} />}
          <section className="replay-objects">
            <div className="replay-section-heading"><h2>Объекты сцены</h2><span>{String((dataset?.tracks.length || 0) + pendingObjects.length).padStart(2, "0")}</span></div>
            {dataset?.tracks.map((item) => (
              <div className={`replay-object-row ${item.id === selectedId ? "is-selected" : ""}`} key={item.id}>
                <button type="button" onClick={() => selectTrack(item.id)} aria-pressed={item.id === selectedId}>
                  <i style={{ background: item.color }} /><span>{item.name}<small>{item.kind === "drone" ? "Квадрокоптер · XYZ из видео" : item.source === "triangulated" ? "Объект · XYZ из видео" : item.source === "estimate" ? "Грузовик · XYZ из видео (offline)" : "3D-модель · исходная траектория"}</small></span>
                  <MotionWaveform track={item} time={time} compact />
                </button>
                <button type="button" className="object-visibility" aria-label={`Показать ${item.name}`} aria-pressed={!hiddenTracks.includes(item.id)} onClick={() => toggleTrack(item.id)}>
                  <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M2 10s3-5 8-5 8 5 8 5-3 5-8 5-8-5-8-5Z" /><circle cx="10" cy="10" r="2" />{hiddenTracks.includes(item.id) && <path d="m3 3 14 14" />}</svg>
                </button>
              </div>
            ))}
            {pendingObjects.map((item) => (
              <div className={`replay-object-row ${item.id === selectedId ? "is-selected" : ""}`} key={item.id}>
                <button type="button" onClick={() => selectTrack(item.id)} aria-pressed={item.id === selectedId}>
                  <i style={{ background: "#69b9ed" }} /><span>{item.name}<small>Наблюдается · XYZ не определены</small></span>
                  <MotionWaveform time={time} compact />
                </button>
              </div>
            ))}
          </section>
          <section className="replay-coordinates">
            <div className="replay-section-heading"><h2>Графики координат</h2><span>ВЫБРАННЫЙ ОБЪЕКТ</span></div>
            <CoordinateChart track={track} time={time} />
          </section>
          <footer className="replay-footer"><span>PELENG / TRAJECTORY LAB</span><span>{stats ? `${stats.calls} draw calls · ${(stats.triangles / 1000).toFixed(1)}K triangles` : "WebGL"}</span></footer>
        </aside>
      </div>
    </main>
  );
}
