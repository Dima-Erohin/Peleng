// Онлайн-прогон (offline/online.py): сервер читает видео камер как живые потоки, считает грузовик
// в момент показа и шлёт точки по WebSocket /ws/online. Здесь — чистые функции для их приёма.
import { createTrack } from "./data.js";

export const ONLINE_TRACK_ID = "truck-online";

// Точки сервера -> трек сцены (Unreal, см -> метры). null, пока точек с координатами меньше двух.
export function buildOnlineTrack(points, { id = ONLINE_TRACK_ID, name = "Peleng · онлайн", color = "#89c7a2" } = {}) {
  const located = points.filter((point) => Array.isArray(point.pos));
  if (located.length < 2) return null;
  // точки идут с шагом обработки (≈0.3–2 с), поэтому разрывом считаем только паузу дольше 5 с
  const track = createTrack({ id, name, units: "cm", maxGap: 5, samples: located.map((point) => ({ t: point.t, position: point.pos })) });
  const last = located.at(-1);
  return {
    ...track, kind: "truck", source: "online", live: true, color,
    liveVel: Array.isArray(last.vel) ? last.vel.map((value) => value / 100) : [0, 0, 0],
  };
}

// Время прогона на сервере (секунды от начала видео) по часам браузера.
export function onlineClock(hello, receivedAt = Date.now() / 1000) {
  const offset = hello.server_time - receivedAt;
  return () => Date.now() / 1000 + offset - hello.t0;
}

// Сводка для панели: частота точек, медианы задержки и времени шага по последним 10 точкам.
export function onlineStats(points) {
  const recent = points.slice(-10);
  const median = (values) => {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
  };
  const span = recent.length > 1 ? recent.at(-1).t - recent[0].t : 0;
  return {
    count: points.length,
    rate: span > 0 ? (recent.length - 1) / span : null,
    lagMs: median(recent.map((point) => point.lag_ms)),
    procMs: median(recent.map((point) => point.proc_ms)),
    last: points.at(-1) || null,
  };
}
