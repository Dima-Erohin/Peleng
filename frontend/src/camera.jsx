import React, { useCallback, useEffect, useRef, useState } from "react";
import "./styles.css";
import { CameraConnectionPanel, CameraPageHeader, CameraSettings } from "./components/index.js";

const RESOLUTIONS = ["1280x720", "1920x1080", "960x540", "640x480"];

export default function Camera() {
  const videoRef = useRef(null);
  const canvasRef = useRef(document.createElement("canvas"));
  const contextRef = useRef(canvasRef.current.getContext("2d"));
  const streamRef = useRef(null);
  const socketRef = useRef(null);
  const runningRef = useRef(false);
  const wakeLockRef = useRef(null);
  const busyRef = useRef(false);
  const pendingWidthRef = useRef(null);
  const offsetRef = useRef(0);
  const bestRttRef = useRef(Infinity);
  const lastRttRef = useRef(0);
  const sentRef = useRef(0);
  const lastSizeRef = useRef("");
  const connectRef = useRef(null);
  const [cameraId, setCameraId] = useState(() => localStorage.getItem("peleng.camId") || `phone${Math.floor(Math.random() * 90 + 10)}`);
  const [resolution, setResolution] = useState(() => {
    const saved = localStorage.getItem("peleng.res");
    return RESOLUTIONS.includes(saved) ? saved : RESOLUTIONS[0];
  });
  const [fps, setFps] = useState(() => localStorage.getItem("peleng.fps") === "60" ? "60" : "30");
  const [devices, setDevices] = useState([]);
  const [selectedDevice, setSelectedDevice] = useState("");
  const [canLock, setCanLock] = useState(false);
  const [focusLocked, setFocusLocked] = useState(false);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("Ожидание запуска");
  const [error, setError] = useState("");

  const listCameras = useCallback(async () => {
    const inputs = (await navigator.mediaDevices.enumerateDevices()).filter((device) => device.kind === "videoinput");
    setDevices(inputs);
    if (!selectedDevice && inputs.length) setSelectedDevice(inputs[0].deviceId);
  }, [selectedDevice]);

  const openCamera = useCallback(async (deviceId) => {
    const videoElement = videoRef.current;
    if (!videoElement) throw new Error("Элемент камеры недоступен");
    const [width, height] = resolution.split("x").map(Number);
    const constraints = {
      width: { ideal: width },
      height: { ideal: height },
      frameRate: { ideal: Number(fps) },
    };
    if (deviceId) constraints.deviceId = { exact: deviceId };
    else constraints.facingMode = { ideal: "environment" };
    streamRef.current?.getTracks().forEach((track) => track.stop());
    const stream = await navigator.mediaDevices.getUserMedia({ video: constraints, audio: false });
    streamRef.current = stream;
    videoElement.srcObject = stream;
    await videoElement.play();
    const capabilities = stream.getVideoTracks()[0].getCapabilities?.() || {};
    setCanLock(Boolean(capabilities.focusMode || capabilities.exposureMode));
    setFocusLocked(false);
    await listCameras();
  }, [fps, listCameras, resolution]);

  const sendFrame = useCallback((width) => {
    const video = videoRef.current;
    const socket = socketRef.current;
    if (!runningRef.current || !socket || socket.readyState !== WebSocket.OPEN ||
        socket.bufferedAmount > 1.5e6 || !video?.videoWidth) return;
    if (busyRef.current) {
      pendingWidthRef.current = width;
      return;
    }

    busyRef.current = true;
    const timestamp = Date.now() / 1000 + offsetRef.current;
    const videoWidth = video.videoWidth;
    const targetWidth = width && width < videoWidth ? width : videoWidth;
    const targetHeight = Math.round(video.videoHeight * targetWidth / videoWidth / 2) * 2;
    const canvas = canvasRef.current;
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
      canvas.width = targetWidth;
      canvas.height = targetHeight;
    }
    contextRef.current.drawImage(video, 0, 0, targetWidth, targetHeight);
    canvas.toBlob(async (blob) => {
      try {
        if (blob && socket.readyState === WebSocket.OPEN) {
          const bytes = new Uint8Array(8 + blob.size);
          new DataView(bytes.buffer).setFloat64(0, timestamp, true);
          bytes.set(new Uint8Array(await blob.arrayBuffer()), 8);
          socket.send(bytes);
          sentRef.current += 1;
          lastSizeRef.current = `${targetWidth}×${targetHeight}`;
        }
      } catch (sendError) {
        setError(`Не удалось отправить кадр: ${sendError.message}`);
      } finally {
        busyRef.current = false;
        if (pendingWidthRef.current !== null) {
          const nextWidth = pendingWidthRef.current;
          pendingWidthRef.current = null;
          sendFrame(nextWidth);
        }
      }
    }, "image/jpeg", 0.75);
  }, []);

  const connect = useCallback(() => {
    const id = cameraId.trim() || "phone";
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${protocol}://${location.host}/ws/push/${encodeURIComponent(id)}`);
    socket.binaryType = "arraybuffer";
    socketRef.current = socket;
    socket.onopen = () => {
      const settings = streamRef.current?.getVideoTracks()[0]?.getSettings() || {};
      socket.send(JSON.stringify({
        type: "hello",
        name: id,
        ua: navigator.userAgent,
        width: settings.width,
        height: settings.height,
        frameRate: settings.frameRate,
        vw: videoRef.current?.videoWidth,
        vh: videoRef.current?.videoHeight,
        on_demand: true,
      }));
      socket.send(JSON.stringify({ type: "ping", tc: Date.now() }));
    };
    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === "capture") {
        sendFrame(message.w || 0);
      } else if (message.type === "pong") {
        const now = Date.now();
        const rtt = now - message.tc;
        if (rtt <= bestRttRef.current * 1.2) {
          bestRttRef.current = Math.min(bestRttRef.current, rtt);
          offsetRef.current = message.ts - (message.tc + rtt / 2) / 1000;
        }
        lastRttRef.current = rtt;
      }
    };
    socket.onclose = () => {
      if (runningRef.current && socketRef.current === socket) {
        setStatus("Переподключение к серверу…");
        window.setTimeout(() => connectRef.current?.(), 1000);
      }
    };
    socket.onerror = () => setError("Ошибка соединения с сервером");
  }, [cameraId, sendFrame]);

  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  useEffect(() => {
    const ping = window.setInterval(() => {
      const socket = socketRef.current;
      if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "ping", tc: Date.now() }));
    }, 2000);
    const fadeRtt = window.setInterval(() => { bestRttRef.current *= 1.05; }, 5000);
    const updateStatus = window.setInterval(() => {
      if (!runningRef.current) return;
      const video = videoRef.current;
      const sent = sentRef.current;
      sentRef.current = 0;
      const connected = socketRef.current?.readyState === WebSocket.OPEN;
      setStatus(`Камера ${video?.videoWidth || 0}×${video?.videoHeight || 0} · ${sent} к/с ${lastSizeRef.current} · RTT ${lastRttRef.current} мс · ${connected ? "подключено" : "переподключение…"}`);
    }, 1000);
    return () => {
      window.clearInterval(ping);
      window.clearInterval(fadeRtt);
      window.clearInterval(updateStatus);
    };
  }, []);

  useEffect(() => {
    const onVisibilityChange = async () => {
      if (document.visibilityState === "visible" && runningRef.current) {
        try { wakeLockRef.current = await navigator.wakeLock?.request("screen"); } catch { /* Unsupported by some browsers */ }
      }
    };
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => document.removeEventListener("visibilitychange", onVisibilityChange);
  }, []);

  useEffect(() => () => {
    runningRef.current = false;
    socketRef.current?.close();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    wakeLockRef.current?.release?.();
  }, []);

  const start = async () => {
    setError("");
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Браузер не даёт доступ к камере. Откройте страницу по HTTPS и примите сертификат.");
      return;
    }
    localStorage.setItem("peleng.camId", cameraId.trim());
    localStorage.setItem("peleng.res", resolution);
    localStorage.setItem("peleng.fps", fps);
    try {
      await openCamera(selectedDevice || null);
      runningRef.current = true;
      setRunning(true);
      setStatus("Подключаем камеру к серверу…");
      connectRef.current?.();
      try { wakeLockRef.current = await navigator.wakeLock?.request("screen"); } catch { /* Unsupported by some browsers */ }
    } catch (startError) {
      setError(`Камера: ${startError.message}`);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  };

  const stop = () => {
    runningRef.current = false;
    pendingWidthRef.current = null;
    socketRef.current?.close();
    socketRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    wakeLockRef.current?.release?.();
    setRunning(false);
    setStatus("Камера остановлена");
  };

  const changeCamera = async (deviceId) => {
    setSelectedDevice(deviceId);
    if (runningRef.current) {
      try {
        await openCamera(deviceId);
        socketRef.current?.close();
        connectRef.current?.();
      } catch (cameraError) {
        setError(`Камера: ${cameraError.message}`);
      }
    }
  };

  const lockFocus = async () => {
    const track = streamRef.current?.getVideoTracks()[0];
    if (!track) return;
    const capabilities = track.getCapabilities?.() || {};
    const advanced = {};
    if (capabilities.focusMode?.includes("manual")) advanced.focusMode = "manual";
    if (capabilities.exposureMode?.includes("manual")) advanced.exposureMode = "manual";
    if (capabilities.whiteBalanceMode?.includes("manual")) advanced.whiteBalanceMode = "manual";
    try {
      await track.applyConstraints({ advanced: [advanced] });
      setFocusLocked(true);
    } catch (lockError) {
      setError(`Не удалось зафиксировать настройки: ${lockError.message}`);
    }
  };

  return (
    <main className="phone-app">
      <video ref={videoRef} className="phone-video" playsInline muted autoPlay />
      <CameraPageHeader running={running} />
      <div className="phone-center-mark"><span /></div>
      <CameraConnectionPanel
        running={running}
        status={status}
        error={error}
        onStart={start}
        onStop={stop}
        canLock={canLock}
        focusLocked={focusLocked}
        onLock={lockFocus}
      >
        <CameraSettings
          cameraId={cameraId}
          resolution={resolution}
          resolutions={RESOLUTIONS}
          fps={fps}
          devices={devices}
          selectedDevice={selectedDevice}
          running={running}
          onCameraId={setCameraId}
          onResolution={setResolution}
          onFps={setFps}
          onChangeCamera={changeCamera}
        />
      </CameraConnectionPanel>
    </main>
  );
}
