import React, { useEffect, useRef, useState } from "react";
import { BrandHeader } from "../BrandHeader";
import { Button } from "../ui";
import { Select } from "../Dropdown";
import PlaybackControls, { PlaybackIcon } from "./PlaybackControls";
import CoordinateChart from "./CoordinateChart";
import CameraVideo from "./CameraVideo";
import { createDataset, fetchText, loadDemo, MAX_TRACKS, MAX_TOTAL_SAMPLES, parseTrajectoryCsv, parseTrajectorySvg } from "../../trajectory/data.js";

const EMPTY_SAMPLE = { x: 0, y: 0, z: 0, speed: 0, distance: 0, visible: false };

export default function TrajectoryWorkspace({ active, navigation }) {
  const containerRef = useRef(null);
  const engineRef = useRef(null);
  const dataInputRef = useRef(null);
  const videoInputRef = useRef(null);
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
  const [selectedId, setSelectedId] = useState("");
  const [hiddenTracks, setHiddenTracks] = useState([]);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [fps, setFps] = useState(30);
  const [units, setUnits] = useState("cm");
  const [url, setUrl] = useState("");
  const track = dataset?.tracks.find((item) => item.id === selectedId);

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
      if (!controller.signal.aborted) { setDataset(value); setClips(value.clips); setLoading(false); }
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

  useEffect(() => { engineRef.current?.setActive(active); }, [active, engineReady]);
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
      if (!controller.signal.aborted) { acceptDataset(value); setFps(30); setUnits("cm"); }
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
  const toggleTrack = (id) => {
    engineRef.current?.setObjectVisible(id, hiddenTracks.includes(id));
    setHiddenTracks((values) => values.includes(id) ? values.filter((value) => value !== id) : [...values, id]);
  };
  const ready = Boolean(dataset && engineReady && modelReady && !loading);

  return (
    <main className="app-shell replay-shell">
      <BrandHeader connected={ready} status={loading ? "Загрузка данных…" : playing ? "Воспроизведение" : "Траектории · пауза"}
        mode="Траектории" detector="CSV / SVG">{navigation}</BrandHeader>
      <div className="replay-workspace">
        <section className="replay-main">
          <div className="replay-scene">
            <div className="replay-canvas" ref={containerRef} />
            <div className="replay-scene-top">
              <div className="scene-title"><span>ПРОСТРАНСТВЕННЫЙ ТРЕКИНГ</span><h1>{dataset?.name || "Загрузка сцены"}</h1>
                <p>{dataset ? `Объектов: ${dataset.tracks.length} · камер: ${dataset.cameras.length} · Z вверх · метры` : "Подготовка траекторий и 3D-модели"}</p></div>
              <span className="replay-data-badge"><i />{dataset?.name === "Project 02" ? "ЭТАЛОННЫЕ ДАННЫЕ" : "ИМПОРТИРОВАННЫЕ ДАННЫЕ"}</span>
            </div>
            <div className="scene-tool-stack" role="group" aria-label="Управление 3D-камерой">
              <button type="button" className={follow ? "is-active" : ""} aria-label="Следовать за объектом" aria-pressed={follow}
                onClick={() => setFollow(!follow)}><PlaybackIcon name="target" /><span>Следовать</span></button>
              <button type="button" aria-label="Вся траектория" onClick={() => { setFollow(false); engineRef.current?.focus(true); }}><PlaybackIcon name="overview" /><span>Обзор</span></button>
              <button type="button" aria-label="Приблизить объект" onClick={() => engineRef.current?.focus()}><PlaybackIcon name="reset" /><span>К объекту</span></button>
            </div>
            {!modelReady && !error && <div className="scene-loading" role="status"><span className="loading-dot" />{loading ? "Читаем траектории" : "Загружаем CargoTruck.fbx"}</div>}
            <div className="replay-scene-bottom">
              <div className="scene-object-tag"><i style={{ background: track?.color }} /><span>{track?.name || "Объект"}<small>{sample.visible ? "Отслеживается" : "Нет отсчёта"}</small></span><b>{sample.visible ? (sample.speed * 3.6).toFixed(1) : "—"}<small>км/ч</small></b></div>
              <div className="render-stats" aria-label="Производительность рендера"><b>{playing ? stats?.fps ?? "—" : "IDLE"}</b> {playing ? "FPS" : "RENDER"}<span>p95 {stats?.frameMs ?? "—"} мс</span></div>
            </div>
            <div className="scene-axis" aria-label="Оси координат"><span>X</span><span>Y</span><span>Z</span></div>
          </div>
          <PlaybackControls dataset={dataset} time={time} playing={playing} rate={rate} loop={loop} disabled={!ready} fps={dataset?.fps || 30}
            onPlay={play} onSeek={seek} onStep={step} onRate={setRate} onLoop={setLoop} />
        </section>
        <aside className="replay-sidebar">
          {error && <div className="replay-error" role="alert"><b>Не удалось выполнить действие</b><span>{error}</span><button type="button" onClick={() => setError("")} aria-label="Закрыть сообщение">×</button></div>}
          <section className="replay-source">
            <div className="replay-section-heading"><h2>Источник данных</h2><span>01</span></div>
            <div className="source-card"><span className="source-file-icon">XYZ</span><div><b>{dataset?.name || "Нет данных"}</b><small>{track ? `Отсчётов: ${track.times.length.toLocaleString("ru")} · ${(dataset.duration).toFixed(2)} с` : "CSV или SVG с временными отметками"}</small></div><i /></div>
            {track?.skippedFrames?.length > 0 && <p className="source-warning" role="status">В исходном CSV повреждены {track.skippedFrames.length} координаты (кадры {track.skippedFrames.join(", ")}). Строки пропущены; между соседними отсчётами показана интерполяция.</p>}
            {dataset?.dataQuality && <div className="source-quality" role="status">
              <b>CSV проверены · UTF-8 · 1002 кадра</b>
              <span>Два значения восстановлены интерполяцией, не измерены. Оригиналы сохранены.</span>
              <a href={`${import.meta.env.BASE_URL}demo/project2/clean/object_transform_SM_AA_track.csv`} download>Скачать корректный CSV</a>
              <a href={`${import.meta.env.BASE_URL}demo/project2/clean/report.json`} target="_blank" rel="noreferrer">Отчёт восстановления и SHA-256</a>
            </div>}
            <div className="source-buttons">
              <Button onClick={() => dataInputRef.current?.click()} disabled={loading}>Загрузить CSV / SVG</Button>
              <Button onClick={() => videoInputRef.current?.click()}>Видео</Button>
            </div>
            <input ref={dataInputRef} type="file" accept=".csv,.svg" multiple hidden onChange={(event) => { loadFiles([...event.target.files]); event.target.value = ""; }} />
            <input ref={videoInputRef} type="file" accept="video/mp4,video/webm" multiple hidden onChange={(event) => { loadVideos([...event.target.files]); event.target.value = ""; }} />
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
          {dataset?.imageTracks?.length > 0 && <section className="replay-image-tracks">
            <div className="replay-section-heading"><h2>Дрон · видеотрекинг</h2><span>2D, НЕ XYZ</span></div>
            <p>Камера 5 · OpenCV · объект указан вручную. Рамка вычислена по изображению; после потери совпадения скрывается.</p>
            <p>{dataset.imageTracks[0].visible_frames} из {dataset.imageTracks[0].frame_count} кадров с совпадением. Откройте CAM 05 · 2D и разверните видео.</p>
          </section>}
          <section className="replay-objects">
            <div className="replay-section-heading"><h2>Объекты сцены</h2><span>{String(dataset?.tracks.length || 0).padStart(2, "0")}</span></div>
            {dataset?.tracks.map((item) => (
              <div className={`replay-object-row ${item.id === selectedId ? "is-selected" : ""}`} key={item.id}>
                <button type="button" onClick={() => selectTrack(item.id)} aria-pressed={item.id === selectedId}>
                  <i style={{ background: item.color }} /><span>{item.name}<small>CargoTruck · 3D-модель</small></span>
                </button>
                <button type="button" className="object-visibility" aria-label={`Показать ${item.name}`} aria-pressed={!hiddenTracks.includes(item.id)} onClick={() => toggleTrack(item.id)}>
                  <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M2 10s3-5 8-5 8 5 8 5-3 5-8 5-8-5-8-5Z" /><circle cx="10" cy="10" r="2" />{hiddenTracks.includes(item.id) && <path d="m3 3 14 14" />}</svg>
                </button>
              </div>
            ))}
          </section>
          <section className="replay-coordinates">
            <div className="replay-section-heading"><h2>Положение объекта</h2><span>МИРОВЫЕ XYZ</span></div>
            <div className="xyz-readout">
              {["x", "y", "z"].map((axis) => <div key={axis}><span>{axis.toUpperCase()}</span><b>{sample.visible ? sample[axis].toFixed(2) : "—"}</b><small>м</small></div>)}
            </div>
            <div className="motion-readout"><span>Скорость<b>{sample.visible ? sample.speed.toFixed(2) : "—"} <small>м/с</small></b></span><span>Пройдено<b>{sample.distance.toFixed(1)} <small>м</small></b></span></div>
            <CoordinateChart track={track} time={time} />
          </section>
          <CameraVideo clips={clips} imageTracks={dataset?.imageTracks || []} time={time} playing={playing} rate={rate} active={active} />
          <footer className="replay-footer"><span>PELENG / TRAJECTORY LAB</span><span>{stats ? `${stats.calls} draw calls · ${(stats.triangles / 1000).toFixed(1)}K triangles` : "WebGL"}</span></footer>
        </aside>
      </div>
    </main>
  );
}
