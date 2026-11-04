import { sampleTrack } from "./data.js";

export function trackingPerformance({ reference, estimate, points, live = false }) {
  const samples = points ?? (!live && estimate ? Array.from(estimate.times, (t) => ({ t })) : []);
  return samples.map((point, index) => {
    let error = null;
    const expected = reference && sampleTrack(reference, point.t);
    const actual = live
      ? Array.isArray(point.pos) && point.pos.length === 3 && point.pos.every(Number.isFinite)
        ? { visible: true, x: point.pos[0] / 100, y: point.pos[1] / 100, z: point.pos[2] / 100 }
        : null
      : estimate && sampleTrack(estimate, point.t);
    if (point.mode !== "none" && expected?.visible && actual?.visible) {
      error = Math.hypot(actual.x - expected.x, actual.y - expected.y, actual.z - expected.z);
    }
    const first = Math.max(0, index - 9);
    const span = point.t - samples[first].t;
    const hz = live && index > first && span > 0 ? (index - first) / span : null;
    return { t: point.t, error, hz };
  });
}

export function metricPath(rows, key, start, end, maximum) {
  let previous = null;
  return rows.map((row) => {
    const value = row[key];
    if (!Number.isFinite(value)) { previous = null; return ""; }
    const command = previous === null || row.t - previous > 1 ? "M" : "L";
    previous = row.t;
    const x = 30 + (row.t - start) / Math.max(0.001, end - start) * 240;
    const y = 116 - value / maximum * 96;
    return `${command}${x.toFixed(2)},${y.toFixed(2)}`;
  }).filter(Boolean).join(" ");
}
