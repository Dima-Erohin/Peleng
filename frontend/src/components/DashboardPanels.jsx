import React from "react";
import { Button, Panel } from "./ui";

export function CalibrationPanel({ showBoard, onShowBoard, onCalibrate }) {
  return (
    <Panel title="Калибровка">
      <Button variant="primary" className="button-wide" onClick={onCalibrate}>
        Калибровать по доске <span>↗</span>
      </Button>
      <p className="calibration-help">Положите доску на пол так, чтобы её видели камеры.</p>
      <a className="text-link" href="/api/board.png" target="_blank" rel="noreferrer">Скачать калибровочную доску ↗</a>
      <label className="check-row">
        <input type="checkbox" checked={showBoard} onChange={(event) => onShowBoard(event.target.checked)} />
        <span>Показывать доску на превью</span>
      </label>
    </Panel>
  );
}

export function SceneControlsPanel({
  showTextures,
  showSkeletons,
  onShowTextures,
  onShowSkeletons,
  onScan,
  onClearRoom,
  onResetTrails,
  onSetMark,
}) {
  return (
    <Panel title="Сцена">
      <div className="button-grid">
        <Button onClick={onScan}>Сканировать комнату</Button>
        <Button onClick={onClearRoom}>Убрать точки</Button>
        <Button onClick={onResetTrails}>Сбросить траектории</Button>
        <Button onClick={onSetMark}>Поставить отметку <kbd>M</kbd></Button>
      </div>
      <label className="check-row">
        <input type="checkbox" checked={showTextures} onChange={(event) => onShowTextures(event.target.checked)} />
        <span>Показывать кадры в 3D</span>
      </label>
      <label className="check-row">
        <input type="checkbox" checked={showSkeletons} onChange={(event) => onShowSkeletons(event.target.checked)} />
        <span>Показывать скелеты</span>
      </label>
    </Panel>
  );
}

export function RecordingPanel({ recording, onToggle }) {
  return (
    <Panel title="Запись">
      <Button variant={recording ? "danger" : "secondary"} className="button-record" onClick={onToggle}>
        {recording && <span className="record-dot" />}
        {recording ? "Остановить запись" : "Записать сессию"}
      </Button>
      {recording && <span className="recording-label">ЗАПИСЬ ИДЁТ</span>}
    </Panel>
  );
}

export function EventLog({ logs }) {
  return (
    <Panel title="Журнал">
      <div id="log" className="log-list">
        {logs.length
          ? logs.map((line, index) => <div key={`${index}-${line}`}>{line}</div>)
          : <span className="log-empty">События появятся здесь.</span>}
      </div>
    </Panel>
  );
}
