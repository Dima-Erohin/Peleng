import { sampleTrack } from "./data.js";

export function motionScale(track) {
  let peak = 0;
  if (!track) return 1;
  for (let i = 1; i < track.times.length; i++) {
    const dt = track.times[i] - track.times[i - 1];
    if (dt > track.maxGap) continue;
    const p = i * 3;
    peak = Math.max(peak, Math.hypot(
      track.positions[p] - track.positions[p - 3],
      track.positions[p + 1] - track.positions[p - 2],
      track.positions[p + 2] - track.positions[p - 1],
    ) / dt);
  }
  return Math.max(peak, 0.01);
}

export function motionWaveform(track, time, scale, count = 48, duration = 4) {
  return Array.from({ length: count }, (_, i) => {
    if (!track) return null;
    const sample = sampleTrack(track, time - duration * (1 - i / (count - 1)));
    return sample.visible ? Math.min(1, sample.speed / scale) : null;
  });
}
