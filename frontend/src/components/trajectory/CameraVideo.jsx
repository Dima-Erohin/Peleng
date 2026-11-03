import React, { useEffect, useRef, useState } from "react";
import { timecode } from "./PlaybackControls";
import ImageTrackOverlay from "./ImageTrackOverlay";

export default function CameraVideo({ clips, imageTracks = [], time, playing, rate, active }) {
  const [selectedId, setSelectedId] = useState(clips[0]?.id);
  const [error, setError] = useState("");
  const [ready, setReady] = useState(false);
  const videoRef = useRef(null);
  const frameRef = useRef(null);
  const currentTime = useRef(time);
  currentTime.current = time;
  const selected = clips.find((clip) => clip.id === selectedId) || clips[0];
  const selectedTracks = imageTracks.filter((track) => track.camera_id === selected?.id);
  useEffect(() => {
    setError("");
    setReady(false);
  }, [selected?.url]);
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !ready || error) return;
    video.playbackRate = rate;
    if (playing && active && !document.hidden) {
      video.play().catch((reason) => {
        if (reason.name !== "AbortError") setError(`Видео не запустилось: ${reason.message}`);
      });
    } else video.pause();
    const visibility = () => {
      if (document.hidden) video.pause();
      else if (playing && active) {
        video.currentTime = Math.min(currentTime.current, Math.max(0, video.duration - 0.01));
        video.play().catch((reason) => {
          if (reason.name !== "AbortError") setError(`Видео: ${reason.message}`);
        });
      }
    };
    document.addEventListener("visibilitychange", visibility);
    return () => { video.pause(); document.removeEventListener("visibilitychange", visibility); };
  }, [playing, rate, active, selected?.url, ready, error]);
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !ready || !Number.isFinite(video.duration)) return;
    const target = Math.min(time, Math.max(0, video.duration - 0.01));
    if (Math.abs(video.currentTime - target) > (playing ? 0.15 : 0.015)) video.currentTime = target;
    if (!error && playing && active && !document.hidden && video.paused && target < video.duration - 0.1) {
      video.play().catch((reason) => {
        if (reason.name !== "AbortError") setError(`Видео: ${reason.message}`);
      });
    }
  }, [time, playing, ready, active, error]);
  return (
    <section className="replay-video-panel" aria-label="Контрольное видео">
      <div className="replay-video-heading">
        <span>КОНТРОЛЬНОЕ ВИДЕО</span><span>{timecode(time)} <i>SYNC</i></span>
      </div>
      {selected ? (
        <>
          <div className="replay-video-frame" ref={frameRef}>
            <video
              key={selected.url} ref={videoRef} src={selected.url} muted playsInline preload="metadata"
              onLoadedMetadata={() => setReady(true)}
              onError={() => setError("Видео недоступно. Загрузите MP4 из архива кнопкой «Видео».")}
              aria-label={selected.name}
            />
            <span className="video-camera-label">{selected.name}</span>
            {selectedTracks.map((track, index) => <ImageTrackOverlay key={track.object_id} track={track}
              videoRef={videoRef} time={time} ready={ready} statusIndex={index} />)}
            <button type="button" className="video-expand" aria-label="Развернуть контрольное видео"
              onClick={() => {
                if (!frameRef.current?.requestFullscreen) { setError("Полноэкранный режим недоступен в этом браузере"); return; }
                frameRef.current.requestFullscreen().catch((reason) => setError(`Полный экран: ${reason.message}`));
              }}>⤢</button>
            {error && <div className="video-message" role="status">{error}</div>}
          </div>
          <div className="video-camera-tabs" role="group" aria-label="Выбор контрольной камеры">
            {clips.map((clip, index) => (
              <button type="button" key={clip.id} className={clip.id === selected.id ? "is-active" : ""} aria-pressed={clip.id === selected.id}
                onClick={() => setSelectedId(clip.id)}><i />CAM {String(index + 1).padStart(2, "0")}{imageTracks.some((track) => track.camera_id === clip.id) ? " · 2D" : ""}</button>
            ))}
          </div>
        </>
      ) : <div className="video-empty">Загрузите контрольное MP4-видео. Одновременно декодируется только выбранная камера.</div>}
    </section>
  );
}
