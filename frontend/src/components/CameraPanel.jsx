import React from "react";
import { Button, Panel, StatusBadge } from "./ui";

function CameraCard({ camera, onIntrinsics, onReset, onHfov, job }) {
  const optics = camera.intrinsics === "board"
    ? { tone: "good", label: "Оптика настроена" }
    : camera.intrinsics === "approx+focal"
      ? { tone: "warning", label: "Фокус уточнён" }
      : { tone: "warning", label: "Оптика приблизительная" };

  return (
    <article className="camera-card">
      <header className="camera-heading">
        <span className={`status-dot ${camera.online ? "online" : ""}`} />
        <span className="camera-name">{camera.name || camera.id}</span>
        <span className="camera-meta">{camera.kind} · {camera.fps || 0} к/с</span>
      </header>
      <div className="camera-tags">
        <StatusBadge tone={camera.calibrated ? "good" : "warning"}>
          {camera.calibrated ? "Положение настроено" : "Нужна калибровка"}
        </StatusBadge>
        <StatusBadge tone={optics.tone}>{optics.label}</StatusBadge>
      </div>
      {camera.center && (
        <div className="camera-world">
          <span>Позиция в сцене</span>
          <b>{camera.center.map((value) => Number(value).toFixed(2)).join(", ")} м</b>
        </div>
      )}
      <div className="camera-actions">
        <Button size="small" onClick={() => onIntrinsics(camera.id)}>Калибровать оптику</Button>
        <Button size="small" onClick={() => onReset(camera.id)}>Сбросить позицию</Button>
      </div>
      <label className="field-inline">
        <span>Угол обзора</span>
        <span className="input-suffix">
          <input
            aria-label={`Угол обзора камеры ${camera.name || camera.id}`}
            type="number"
            min="30"
            max="120"
            step="1"
            defaultValue={camera.hfov_deg || 68}
            onBlur={(event) => {
              const value = Number(event.target.value);
              if (value >= 30 && value <= 120) onHfov(camera.id, value);
            }}
          />
          <span>°</span>
        </span>
      </label>
      {job && <div className="camera-job">{job}</div>}
    </article>
  );
}

export default function CameraPanel({ cameras, jobs, onIntrinsics, onReset, onHfov }) {
  return (
    <Panel title="Камеры" className="cameras-section" meta={cameras.length.toString().padStart(2, "0")}>
      {cameras.length ? (
        <div className="camera-list">
          {cameras.map((camera) => (
            <CameraCard
              key={camera.id}
              camera={camera}
              job={jobs[camera.id]}
              onIntrinsics={onIntrinsics}
              onReset={onReset}
              onHfov={onHfov}
            />
          ))}
        </div>
      ) : (
        <div className="empty-state">Пока нет подключённых камер.</div>
      )}
      <a className="phone-link" href="/cam">Открыть на телефоне <b>{location.origin}/cam</b></a>
    </Panel>
  );
}
