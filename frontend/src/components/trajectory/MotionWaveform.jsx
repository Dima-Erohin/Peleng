import React, { useMemo } from "react";
import { motionScale, motionWaveform } from "../../trajectory/motion-waveform.js";

export default function MotionWaveform({ track, time, compact = false }) {
  const scale = useMemo(() => motionScale(track), [track]);
  const values = useMemo(() => motionWaveform(track, time, scale, compact ? 20 : 48), [track, time, scale, compact]);
  const width = compact ? 60 : 240;
  const height = compact ? 16 : 22;
  return (
    <svg className={`motion-waveform${compact ? " is-compact" : ""}`} viewBox={`0 0 ${width} ${height}`}
      role="img" aria-label={track ? `Движение ${track.name}: скорость за последние 4 секунды` : "Движение: нет координат"}
      style={{ color: track?.color || "#69b9ed" }}>
      <title>Визуализатор скорости, не частота в Гц. Высота относительно максимума скорости объекта.</title>
      <path d={`M0 ${height / 2}H${width}`} stroke="currentColor" opacity=".16" />
      {values.map((value, i) => {
        if (value === null) return null;
        const x = 2 + i * (width - 4) / (values.length - 1);
        const amplitude = Math.max(0.7, value * (height / 2 - 3));
        return <path key={i} d={`M${x} ${height / 2 - amplitude}v${amplitude * 2}`}
          stroke="currentColor" strokeWidth={compact ? 1 : 1.25} vectorEffect="non-scaling-stroke"
          strokeLinecap="round" opacity={0.45 + 0.55 * i / (values.length - 1)} />;
      })}
    </svg>
  );
}
