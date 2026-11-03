import React from "react";
import { Button, StatusBadge } from "./ui";

export function CameraPageHeader({ running }) {
  return (
    <header className="phone-brand">
      <img className="phone-brand-logo" src="/static/build/brand/peleng-logo-red.svg" alt="ПЕЛЕНГ" />
      <span className="phone-brand-caption">КАМЕРА</span>
      <StatusBadge tone={running ? "live" : "neutral"}>{running ? "LIVE" : "OFFLINE"}</StatusBadge>
    </header>
  );
}

function CameraControls({ running, canLock, focusLocked, onStart, onStop, onLock }) {
  return (
    <div className="phone-actions">
      {!running ? (
        <Button variant="primary" className="phone-start" onClick={onStart}>Запустить камеру <span>→</span></Button>
      ) : (
        <Button variant="danger" className="phone-stop" onClick={onStop}>Остановить передачу</Button>
      )}
      {running && canLock && (
        <Button className="phone-lock" onClick={onLock} disabled={focusLocked}>
          {focusLocked ? "Фокус зафиксирован" : "Зафиксировать фокус"}
        </Button>
      )}
    </div>
  );
}

export default function CameraConnectionPanel({
  running,
  status,
  error,
  children,
  onStart,
  onStop,
  canLock,
  focusLocked,
  onLock,
}) {
  return (
    <section className="phone-panel">
      <div className="phone-panel-heading">
        <div><span className="phone-eyebrow">КАМЕРА СБОРКИ</span><h1>Подключение камеры</h1></div>
        <span className={`phone-status-dot ${running ? "active" : ""}`} />
      </div>
      {children}
      {error && <div className="phone-error" role="alert">{error}</div>}
      <CameraControls
        running={running}
        canLock={canLock}
        focusLocked={focusLocked}
        onStart={onStart}
        onStop={onStop}
        onLock={onLock}
      />
      <div className="phone-status">{running && <i />}{status}</div>
      <div className="phone-note">Оставьте страницу открытой, пока камера передаёт изображение.</div>
    </section>
  );
}
