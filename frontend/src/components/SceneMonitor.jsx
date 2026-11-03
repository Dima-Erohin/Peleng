import React from "react";
import { Dropdown } from "./Dropdown";
import { StatusBadge } from "./ui";

function Metric({ label, value, unit, accent }) {
  return (
    <div className="scene-metric">
      <span className="metric-label">{label}</span>
      <span className={`metric-value ${accent || ""}`}>
        {value ?? "—"}{value != null && unit && <small>{unit}</small>}
      </span>
    </div>
  );
}

function TrackingPopover({ sceneNotice, telemetry, people }) {
  return (
    <>
      {sceneNotice ? (
        <span className="scene-notice">{sceneNotice}</span>
      ) : telemetry.usedCameras?.length ? (
        <div className="tracking-cameras">
          Камеры участвуют в 3D
          <b>{telemetry.usedCameras.join(" · ")}</b>
        </div>
      ) : (
        <span className="scene-notice">Ожидание данных от камер</span>
      )}
      {people.map((person) => (
        <div className="track-row" key={person.id}>
          <span className="track-id">ID {person.id}</span>
          <span>{person.pos.map((value) => Number(value).toFixed(2)).join(", ")} м</span>
          <span className="track-mode">{person.mode}</span>
        </div>
      ))}
    </>
  );
}

export default function SceneMonitor({ connected, telemetry, people, sceneNotice, showMeasure, measure }) {
  return (
    <div className="scene-wrap">
      <div id="view" className="scene-view" />
      <div className="scene-overlay">
        <StatusBadge tone={connected ? "live" : "neutral"}>
          {connected ? "Сцена активна" : "Ожидание"}
        </StatusBadge>
        <span className="scene-hint">Вращение — перетаскивание · масштаб — колесо</span>
      </div>
      <div className="scene-foot">
        <div className="telemetry">
          <Metric label="Обработка" value={telemetry.fps == null ? null : Number(telemetry.fps).toFixed(1)} unit="кадр/с" accent="mint" />
          <Metric label="Задержка" value={telemetry.latency == null ? null : Number(telemetry.latency).toFixed(0)} unit="мс" />
          <Metric label="Синхронизация" value={telemetry.skew == null ? null : Number(telemetry.skew).toFixed(0)} unit="мс" />
          <Metric label="Детектор" value={telemetry.detect == null ? null : Number(telemetry.detect).toFixed(1)} unit="мс" />
        </div>
        <Dropdown
          label="Информация о треках"
          className="tracks-dropdown"
          popupRole="dialog"
          trigger={(
            <button type="button" className="tracks-trigger">
              <span className="tracks-summary-label">В сцене</span>
              <span className="tracks-summary-count">{people.length.toString().padStart(2, "0")}</span>
              <span className="tracks-summary-text">{people.length === 1 ? "человек" : "людей"}</span>
              <svg className="tracks-chevron" viewBox="0 0 16 16" aria-hidden="true"><path d="m3.5 6 4.5 4 4.5-4" /></svg>
            </button>
          )}
        >
          <div className="tracks-popover">
            <TrackingPopover sceneNotice={sceneNotice} telemetry={telemetry} people={people} />
          </div>
        </Dropdown>
      </div>
      {showMeasure && <div id="measure" className="measure-chip">{measure}</div>}
    </div>
  );
}
