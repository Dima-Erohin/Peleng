import React, { useMemo } from "react";

const AXES = [{ name: "X", color: "#f26b5e" }, { name: "Y", color: "#78b8e4" }, { name: "Z", color: "#d9ba76" }];

export default function CoordinateChart({ track, time }) {
  const series = useMemo(() => {
    if (!track) return [];
    const count = track.times.length;
    const duration = track.times.at(-1) - track.times[0];
    return AXES.map((axis, a) => {
      const min = track.bounds.min[a], max = track.bounds.max[a], span = Math.max(1, max - min);
      const stride = Math.max(1, Math.ceil(count / 400));
      const values = [];
      let pendingGap = true;
      for (let i = 0; i < count; i++) {
        const gap = i > 0 && track.times[i] - track.times[i - 1] > track.maxGap;
        if (gap) pendingGap = true;
        if (i % stride !== 0 && i !== count - 1 && !pendingGap) continue;
        const x = (track.times[i] - track.times[0]) / duration * 260;
        const y = 33 - (track.positions[i * 3 + a] - min) / span * 26;
        values.push(`${pendingGap ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`);
        pendingGap = false;
      }
      return { ...axis, path: values.join(" "), min, max };
    });
  }, [track]);
  if (!track) return null;
  const progress = Math.max(0, Math.min(1, (time - track.times[0]) / (track.times.at(-1) - track.times[0])));
  return (
    <div className="coordinate-charts">
      {series.map((axis) => (
        <div className="coordinate-chart" key={axis.name}>
          <div className="chart-caption"><b style={{ color: axis.color }}>{axis.name}</b><span>{axis.min.toFixed(1)} — {axis.max.toFixed(1)} м</span></div>
          <svg viewBox="0 0 260 40" role="img" aria-label={`${axis.name}: координата от времени`}>
            <path d="M0 36H260M0 18H260" stroke="#293139" strokeWidth="0.5" />
            <path d={axis.path} fill="none" stroke={axis.color} strokeWidth="1.5" />
            <path d={`M${progress * 260} 0v40`} stroke="#e5edf1" strokeWidth="0.8" />
          </svg>
        </div>
      ))}
      <div className="chart-time"><span>0 с</span><span>{(track.times.at(-1) - track.times[0]).toFixed(2)} с</span></div>
    </div>
  );
}
