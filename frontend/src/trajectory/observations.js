export function parseTruckObservations(value) {
  const invalid = () => { throw new Error("Наблюдения грузовика: неверный формат кадров, камер или детекций"); };
  if (!value || !Number.isFinite(value.fps) || value.fps <= 0 ||
      !Array.isArray(value.cams) || !value.cams.length || new Set(value.cams).size !== value.cams.length ||
      !Array.isArray(value.sizes) || value.sizes.length !== value.cams.length ||
      !value.sizes.every((size) => Array.isArray(size) && size.length === 2 && size.every((n) => Number.isInteger(n) && n > 0)) ||
      !Array.isArray(value.points)) invalid();
  let previous = -1;
  for (const point of value.points) {
    if (!point || !Number.isInteger(point.frame) || point.frame <= previous ||
        !Number.isFinite(point.t) || Math.abs(point.t - point.frame / value.fps) > 1e-5 ||
        typeof point.mode !== "string" || !point.mode ||
        !Number.isInteger(point.n_cams) || point.n_cams < 0 || point.n_cams > value.cams.length ||
        typeof point.limited_update !== "boolean" ||
        (point.sigma_m !== null && (!Number.isFinite(point.sigma_m) || point.sigma_m < 0)) ||
        !Array.isArray(point.dets) || point.dets.length !== value.cams.length ||
        !point.dets.every((row) => Array.isArray(row) && row.every((box) =>
          Array.isArray(box) && box.length >= 6 && box.slice(0, 5).every(Number.isFinite) &&
          box[2] > box[0] && box[3] > box[1] && box[4] >= 0 && box[4] <= 1)) ||
        !Array.isArray(point.used_det) || !point.used_det.every((pair) =>
          Array.isArray(pair) && pair.length === 2 && pair.every(Number.isInteger) &&
          pair[0] >= 0 && pair[0] < value.cams.length && pair[1] >= 0 && pair[1] < point.dets[pair[0]].length)) invalid();
    previous = point.frame;
  }
  return value;
}

export async function observationsMatchTrajectory(observations, csv) {
  if (!observations?.trajectory_sha256 || !csv) return false;
  const bytes = new TextEncoder().encode(csv.replace(/\r\n/g, "\n"));
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("") === observations.trajectory_sha256;
}

export function observationAt(points, time) {
  let low = 0, high = points.length - 1, index = -1;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    if (points[middle].t <= time + 1e-6) { index = middle; low = middle + 1; }
    else high = middle - 1;
  }
  return index < 0 ? null : points[index];
}

export function trackingQuality(point, time) {
  if (!point || point.mode === "none") return { level: "lost", label: "Нет надёжных XYZ", detail: "Ждём наблюдения камер" };
  const age = Math.max(0, time - point.t);
  if (age > 1) return { level: "lost", label: "Данные устарели", detail: `${age.toFixed(1)} с без новой оценки` };
  if (point.limited_update) return { level: "warning", label: "Восстановление трека", detail: "Коррекция ограничена · положение уточняется" };
  if (point.mode === "predict") return { level: "warning", label: "Прогноз", detail: "Объект не измерен · движение по модели" };
  if (point.mode.startsWith("mono") || point.n_cams === 1) return {
    level: "warning", label: "Один ракурс", detail: "Глубина ненадёжна · используется прогноз",
  };
  return { level: "measured", label: point.mode === "init" ? "Начало трека" : "Измерение", detail: `${point.n_cams} камеры · многокамерная оценка` };
}

export function frameBoxes(point, cameraIndex, videoTime, fps) {
  if (!point || Math.abs(videoTime - point.t) > 0.6 / fps || cameraIndex < 0) return [];
  return (point.dets?.[cameraIndex] || []).filter((box) =>
    box.length >= 6 && box.slice(0, 5).every(Number.isFinite) &&
    box[0] >= 0 && box[1] >= 0 && box[2] > box[0] && box[3] > box[1] &&
    box[4] >= 0 && box[4] <= 1);
}
