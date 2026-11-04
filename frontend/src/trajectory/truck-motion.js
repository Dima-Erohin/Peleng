import { createTrack } from "./data.js";

// Presentation only: camera measurements and exported coordinates remain unchanged.
export function smoothTruckTrack(track) {
  const { times, positions, maxGap } = track;
  const samples = [];
  const headings = new Float64Array(times.length);
  let start = 0;
  while (start < times.length) {
    let end = start + 1;
    while (end < times.length && times[end] - times[end - 1] <= maxGap) end++;
    let left = start, right = start;
    for (let i = start; i < end; i++) {
      while (times[i] - times[left] > 0.3) left++;
      while (right + 1 < end && times[right + 1] - times[i] <= 0.3) right++;
      const points = [];
      for (let j = left; j <= right; j++) {
        const dt = times[j] - times[i];
        points.push({ j, dt, weight: 1 - Math.abs(dt) / 0.31, robust: 1 });
      }
      let position = Array.from(positions.subarray(i * 3, i * 3 + 3));
      let velocity = [0, 0, 0];
      for (let pass = 0; pass < 3; pass++) {
        let w = 0, t = 0, tt = 0;
        const p = [0, 0, 0], tp = [0, 0, 0];
        for (const point of points) {
          const weight = point.weight * point.robust;
          w += weight; t += weight * point.dt; tt += weight * point.dt ** 2;
          for (let axis = 0; axis < 3; axis++) {
            const value = positions[point.j * 3 + axis];
            p[axis] += weight * value; tp[axis] += weight * point.dt * value;
          }
        }
        const det = w * tt - t * t;
        if (det <= 1e-12) break;
        position = p.map((value, axis) => (value * tt - tp[axis] * t) / det);
        velocity = tp.map((value, axis) => (value * w - p[axis] * t) / det);
        const residuals = points.map((point) => Math.hypot(...position.map((value, axis) =>
          positions[point.j * 3 + axis] - value - velocity[axis] * point.dt)));
        const sorted = [...residuals].sort((a, b) => a - b);
        const threshold = Math.max(0.1, 2 * sorted[Math.floor(sorted.length / 2)]);
        points.forEach((point, index) => { point.robust = Math.min(1, threshold / Math.max(residuals[index], 1e-9)); });
      }
      samples.push({ t: times[i], position });
      const desired = Math.atan2(velocity[1], velocity[0]);
      const moving = Math.hypot(velocity[0], velocity[1]) >= 0.5;
      if (i === start) headings[i] = moving ? desired : track.headings[i];
      else {
        const previous = headings[i - 1];
        const turn = Math.atan2(Math.sin(desired - previous), Math.cos(desired - previous));
        const limit = Math.PI / 2 * (times[i] - times[i - 1]);
        headings[i] = previous + (moving ? Math.max(-limit, Math.min(limit, turn)) : 0);
      }
    }
    start = end;
  }
  return { ...track, ...createTrack({ id: track.id, name: track.name, samples, units: "m", maxGap }), headings };
}
