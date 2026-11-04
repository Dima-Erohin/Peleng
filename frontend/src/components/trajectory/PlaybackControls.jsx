import React from "react";
import { Select } from "../Dropdown";

export function timecode(seconds) {
  const safe = Math.max(0, seconds);
  return `${Math.floor(safe / 60).toString().padStart(2, "0")}:${(safe % 60).toFixed(2).padStart(5, "0")}`;
}

export function PlaybackIcon({ name }) {
  const paths = {
    play: <path d="m6 3 12 7-12 7Z" />,
    pause: <><path d="M6 3h3v14H6zM13 3h3v14h-3z" /></>,
    reset: <><path d="M4 5v5h5" /><path d="M4 10a6 6 0 1 1 1.6 4" /></>,
    previous: <><path d="M5 4v12M16 4 7 10l9 6Z" /></>,
    next: <><path d="M15 4v12M4 4l9 6-9 6Z" /></>,
    loop: <><path d="M3 8V5h12l-3-3M17 12v3H5l3 3" /></>,
    target: <><circle cx="10" cy="10" r="5" /><path d="M10 1v5M10 14v5M1 10h5M14 10h5" /></>,
    overview: <><path d="m10 3 8 4-8 4-8-4ZM2 11l8 4 8-4M2 15l8 4 8-4" /></>,
    route: <><circle cx="4" cy="15" r="2" /><circle cx="16" cy="5" r="2" /><path d="M4 13V8a3 3 0 0 1 3-3h3a3 3 0 0 1 0 6H9a3 3 0 0 0 0 6h4a3 3 0 0 0 3-3V7" /></>,
  };
  return <svg viewBox="0 0 20 20" aria-hidden="true">{paths[name]}</svg>;
}

export default function PlaybackControls({ dataset, time, playing, rate, loop, disabled, onPlay, onSeek, onStep, onRate, onLoop, fps }) {
  const start = dataset?.start || 0, end = dataset?.end || 1;
  return (
    <section className="playback-bar" aria-label="Управление воспроизведением">
      <div className="playback-timeline">
        <span>{timecode(time - start)}</span>
        <input
          aria-label="Позиция воспроизведения"
          type="range" min={start} max={end} step={0.001} value={time} disabled={disabled}
          onChange={(event) => onSeek(Number(event.target.value))}
        />
        <span className="time-total">{timecode(end - start)}</span>
      </div>
      <div className="playback-actions">
        <div className="playback-transport">
          <button type="button" className="transport-button" aria-label="В начало" disabled={disabled} onClick={() => onSeek(start)}><PlaybackIcon name="reset" /></button>
          <button type="button" className="transport-button" aria-label="Предыдущий кадр" disabled={disabled} onClick={() => onStep(-1)}><PlaybackIcon name="previous" /></button>
          <button type="button" className="transport-play" aria-label={playing ? "Пауза" : "Воспроизвести"} disabled={disabled} onClick={onPlay}><PlaybackIcon name={playing ? "pause" : "play"} /></button>
          <button type="button" className="transport-button" aria-label="Следующий кадр" disabled={disabled} onClick={() => onStep(1)}><PlaybackIcon name="next" /></button>
          <button type="button" className={`transport-button ${loop ? "is-active" : ""}`} aria-label="Зациклить воспроизведение" aria-pressed={loop} disabled={disabled} onClick={() => onLoop(!loop)}><PlaybackIcon name="loop" /></button>
        </div>
        <span className="playback-frame">КАДР <b>{Math.round(time * fps).toString().padStart(4, "0")}</b></span>
        <div className="playback-rate">
          <Select label="Скорость воспроизведения" value={String(rate)} disabled={disabled} onChange={(value) => onRate(Number(value))}
            options={[0.25, 0.5, 1, 2].map((value) => ({ value: String(value), label: `${value}×` }))} />
        </div>
      </div>
    </section>
  );
}
