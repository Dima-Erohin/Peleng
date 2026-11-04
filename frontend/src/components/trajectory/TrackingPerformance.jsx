import React, { useMemo } from "react";
import { metricPath, trackingPerformance } from "../../trajectory/tracking-performance.js";

export default function TrackingPerformance({ dataset, points, live, time }) {
  const reference = dataset?.tracks.find((track) => track.kind === "truck" && track.source === "reference");
  const estimate = dataset?.tracks.find((track) => track.kind === "truck" && track.source === "estimate");
  const observations = live ? points : undefined;
  const rows = useMemo(() => trackingPerformance({ reference, estimate, points: observations, live }),
    [reference, estimate, observations, live]);
  const visible = rows.filter((row) => row.t <= time + 1e-6);
  const last = visible.at(-1);
  const start = dataset?.start ?? 0;
  const end = dataset?.end ?? 1;
  const errorMax = Math.max(1, Math.ceil(Math.max(0, ...visible.map((row) => row.error ?? 0)) / 5) * 5);
  const hzMax = Math.max(5, Math.ceil(Math.max(0, ...visible.map((row) => row.hz ?? 0)) / 5) * 5);
  const cursor = 30 + Math.max(0, Math.min(1, (time - start) / Math.max(0.001, end - start))) * 240;
  const stale = last && time - last.t > 1;
  const value = (key, digits) => !stale && Number.isFinite(last?.[key]) ? last[key].toFixed(digits) : "—";
  return (
    <section className="tracking-performance" aria-label="Ошибка XYZ и частота вычислений">
      <div className="replay-section-heading"><h2>Отклонение и частота</h2><span>{live ? "ОНЛАЙН" : "ЗАПИСЬ"}</span></div>
      <div className="performance-readouts">
        <div className="performance-error"><span><i />Ошибка XYZ</span><b>{value("error", 2)} <small>м</small></b></div>
        <div className="performance-hz"><span><i />Частота расчёта</span><b>{value("hz", 1)} <small>Гц</small></b></div>
      </div>
      <svg viewBox="0 0 300 145" role="img" aria-label={live ? "Ошибка XYZ в метрах слева, частота вычислений в герцах справа, по времени" : "Ошибка XYZ в метрах по времени"}>
        <text x="0" y="12" className="performance-error">м</text>
        {live && <text x="300" y="12" textAnchor="end" className="performance-hz">Гц</text>}
        {[0, 0.5, 1].map((fraction) => <g key={fraction}>
          <path d={`M30 ${116 - fraction * 96}H270`} stroke="#34414b" strokeWidth="0.5" />
          <text x="23" y={119 - fraction * 96} textAnchor="end" className="performance-error">{(errorMax * fraction).toFixed(1)}</text>
          {live && <text x="277" y={119 - fraction * 96} className="performance-hz">{(hzMax * fraction).toFixed(1)}</text>}
        </g>)}
        <path className="performance-error-line" d={metricPath(visible, "error", start, end, errorMax)} />
        {live && <path className="performance-hz-line" d={metricPath(visible, "hz", start, end, hzMax)} />}
        <path d={`M${cursor} 20V116`} stroke="#ccd5db" strokeWidth="0.7" strokeDasharray="2 3" />
        <text x="30" y="137">{start.toFixed(1)} с</text>
        <text x="270" y="137" textAnchor="end">{end.toFixed(1)} с</text>
      </svg>
      <p>{live ? "Золотой — ошибка, м · голубой — результаты/с. Шкалы независимы, Гц ≠ FPS." : "Ошибка до эталона · Гц доступны в онлайн-прогоне, не в CSV."}</p>
      {!reference && <p role="status">Нет эталона для сравнения XYZ.</p>}
    </section>
  );
}
