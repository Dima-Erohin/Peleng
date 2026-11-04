import React from "react";
import { Button } from "../ui";

export default function ReconstructionPanel({ reconstructions = [], selectedId, time, onSelect, onImport, disabled }) {
  return (
    <section className="replay-reconstruction">
      <div className="replay-section-heading"><h2>Восстановление по видео</h2><span>2D → 3D</span></div>
      {reconstructions.map((object) => {
        const frame = object.frames[Math.round(time * object.fps)];
        const status = frame?.status === "triangulated" ? "XYZ вычислены" : frame?.status === "rejected"
          ? "Измерения не согласованы" : frame?.cameras.length ? "Не хватает второго ракурса" : "Нет детекций";
        return (
          <div className={`reconstruction-card ${selectedId === object.id ? "is-selected" : ""}`} key={object.id}>
            <button type="button" aria-pressed={selectedId === object.id} onClick={() => onSelect(object.id)}>
              <span>{object.name}</span><small>{object.cameraIds.length} камер · {object.validCount}/{object.frames.length} XYZ</small>
            </button>
            <p role="status">{status}</p>
            {frame?.position
              ? <small>Репроекция: {frame.reprojectionPx.toFixed(2)} px · угол лучей: {frame.angleDeg.toFixed(1)}°</small>
              : <small>На сцене — луч наблюдения, не положение объекта. Дальность неизвестна.</small>}
          </div>
        );
      })}
      <p>Совпадающий ID объекта, синхронные кадры и калибровка минимум двух камер дают XYZ. Выбросы и вырожденная геометрия отклоняются.</p>
      <Button onClick={onImport} disabled={disabled}>Загрузить наблюдения JSON</Button>
      <p>Ошибка репроекции — согласованность изображений, не точность в метрах. Размер модели дрона условный.</p>
    </section>
  );
}
