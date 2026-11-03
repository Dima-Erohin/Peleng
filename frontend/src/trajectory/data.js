const UNIT_SCALE = { m: 1, cm: 0.01, mm: 0.001 };
export const MAX_SAMPLES = 200_000;
export const MAX_TRACKS = 32;
export const MAX_TOTAL_SAMPLES = 500_000;
export const TRACK_COLORS = ["#f26b5e", "#69b9ed", "#d9ba76", "#89c7a2", "#b99bde", "#73c7c7"];

function number(value, description) {
  if (String(value ?? "").trim() === "") throw new Error(`${description}: пустое значение`);
  const parsed = Number(String(value).replace(",", "."));
  if (!Number.isFinite(parsed)) throw new Error(`${description}: требуется конечное число`);
  return parsed;
}

export function createTrack({ id, name, samples, units, maxGap = 0.5 }) {
  if (typeof id !== "string" || !id.trim()) throw new Error("У объекта должен быть непустой ID");
  const scale = UNIT_SCALE[units];
  if (!scale) throw new Error("Единицы координат: укажите m, cm или mm");
  if (samples.length < 2 || samples.length > MAX_SAMPLES) {
    throw new Error(`В траектории должно быть от 2 до ${MAX_SAMPLES} отсчётов`);
  }
  const times = new Float64Array(samples.length);
  const positions = new Float64Array(samples.length * 3);
  const distances = new Float64Array(samples.length);
  const headings = new Float64Array(samples.length);
  const bounds = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
  for (let i = 0; i < samples.length; i++) {
    const sample = samples[i];
    times[i] = number(sample.t, `Отсчёт ${i + 1}, время`);
    if (times[i] < 0 || (i > 0 && times[i] <= times[i - 1])) {
      throw new Error(`Отсчёт ${i + 1}: время должно неотрицательно и строго возрастать`);
    }
    for (let axis = 0; axis < 3; axis++) {
      const value = number(sample.position[axis], `Отсчёт ${i + 1}, XYZ` ) * scale;
      if (Math.abs(value) > 1e7) throw new Error("Координата слишком велика: проверьте единицы");
      positions[i * 3 + axis] = value;
      bounds.min[axis] = Math.min(bounds.min[axis], value);
      bounds.max[axis] = Math.max(bounds.max[axis], value);
    }
    if (i > 0) {
      const offset = i * 3;
      distances[i] = distances[i - 1];
      if (times[i] - times[i - 1] <= maxGap) {
        distances[i] += Math.hypot(
          positions[offset] - positions[offset - 3],
          positions[offset + 1] - positions[offset - 2],
          positions[offset + 2] - positions[offset - 1],
        );
      }
    }
  }
  let heading = 0;
  for (let i = 0; i < samples.length - 1; i++) {
    const dx = positions[(i + 1) * 3] - positions[i * 3];
    const dy = positions[(i + 1) * 3 + 1] - positions[i * 3 + 1];
    if (times[i + 1] - times[i] <= maxGap && Math.hypot(dx, dy) > 0.00001) heading = Math.atan2(dy, dx);
    headings[i] = heading;
  }
  headings[headings.length - 1] = heading;
  return { id: String(id), name: name || String(id), times, positions, distances, headings, bounds, maxGap };
}

export function parseTrajectoryCsv(text, { fps = 30, units = "cm", id = "object-1", name = id, skipInvalidPositions = false } = {}) {
  if (!Number.isFinite(fps) || fps <= 0 || fps > 240) throw new Error("Частота должна быть от 0 до 240 кадров/с");
  const rows = text.replace(/^\uFEFF/, "").trim().split(/\r?\n/);
  const separator = rows[0].includes(";") ? ";" : ",";
  const header = rows[0].split(separator).map((cell) => cell.trim());
  const columns = ["Frame", "Loc_X", "Loc_Y", "Loc_Z"].map((key) => header.indexOf(key));
  if (columns.some((index) => index < 0)) throw new Error("CSV: нужны столбцы Frame;Loc_X;Loc_Y;Loc_Z");
  if (header.includes("Rot_Yaw")) throw new Error("Это CSV камеры. Выберите CSV движения объекта");
  if (rows.length - 1 > MAX_SAMPLES) throw new Error(`CSV: максимум ${MAX_SAMPLES} строк`);
  const skippedFrames = [];
  const samples = [];
  rows.slice(1).forEach((row, i) => {
    const cells = row.split(separator);
    const frame = number(cells[columns[0]], `Строка ${i + 2}, Frame`);
    if (skipInvalidPositions && columns.slice(1).some((column) => {
      const value = cells[column]?.trim();
      return !value || !Number.isFinite(Number(value.replace(",", ".")));
    })) {
      skippedFrames.push(frame);
      return;
    }
    samples.push({
      t: frame / fps,
      position: columns.slice(1).map((column) => number(cells[column], `Строка ${i + 2}, координата`)),
    });
  });
  return { ...createTrack({ id, name, samples, units }), skippedFrames };
}

// SVG is a data container, never inserted into the page or evaluated as markup.
export function parseTrajectorySvg(text, name = "SVG") {
  const document = new DOMParser().parseFromString(text, "image/svg+xml");
  if (document.querySelector("parsererror") || document.documentElement.localName !== "svg") {
    throw new Error("Не удалось прочитать SVG");
  }
  const root = document.documentElement;
  const units = root.getAttribute("data-units");
  if (!UNIT_SCALE[units]) throw new Error("SVG: требуется data-units=\"m\", \"cm\" или \"mm\"");
  if (root.querySelector("[transform]") || root.hasAttribute("transform")) {
    throw new Error("SVG: преобразуйте transform в мировые координаты перед экспортом");
  }
  const paths = [...root.querySelectorAll("polyline[data-track-id]")];
  if (!paths.length || paths.length > MAX_TRACKS) {
    throw new Error(`SVG: требуется от 1 до ${MAX_TRACKS} polyline с data-track-id, data-times и data-z`);
  }
  return paths.map((path) => {
    const coords = (path.getAttribute("points") || "").trim().split(/[\s,]+/);
    const times = (path.getAttribute("data-times") || "").trim().split(/\s+/);
    const z = (path.getAttribute("data-z") || "").trim().split(/\s+/);
    if (coords.length !== times.length * 2 || z.length !== times.length) {
      throw new Error("SVG: каждому XY в points нужны время в data-times и высота в data-z");
    }
    const id = path.getAttribute("data-track-id");
    return createTrack({
      id, name: path.getAttribute("data-name") || `${name} · ${id}`, units,
      samples: times.map((t, i) => ({ t, position: [coords[i * 2], coords[i * 2 + 1], z[i]] })),
    });
  });
}

export function createDataset(tracks, { name = "Траектории", cameras = [], clips = [], fps = 30, imageTracks = [], dataQuality = null } = {}) {
  if (!Number.isFinite(fps) || fps <= 0 || fps > 240) throw new Error("Частота должна быть от 0 до 240 кадров/с");
  if (!tracks.length || tracks.length > MAX_TRACKS) throw new Error(`Загрузите от 1 до ${MAX_TRACKS} траекторий`);
  if (tracks.reduce((sum, track) => sum + track.times.length, 0) > MAX_TOTAL_SAMPLES) throw new Error(`Максимум ${MAX_TOTAL_SAMPLES} отсчётов во всей сцене`);
  if (new Set(tracks.map((track) => track.id)).size !== tracks.length) {
    throw new Error("ID объектов должны быть уникальными");
  }
  const start = Math.min(...tracks.map((track) => track.times[0]));
  const end = Math.max(...tracks.map((track) => track.times.at(-1)));
  const bounds = {
    min: [0, 1, 2].map((axis) => Math.min(...tracks.map((track) => track.bounds.min[axis]))),
    max: [0, 1, 2].map((axis) => Math.max(...tracks.map((track) => track.bounds.max[axis]))),
  };
  return {
    name, start, end, duration: end - start, bounds, cameras, clips, fps, imageTracks, dataQuality,
    tracks: tracks.map((track, i) => ({ ...track, color: TRACK_COLORS[i % TRACK_COLORS.length] })),
  };
}

export function lowerSample(times, time) {
  let low = 0, high = times.length - 1;
  while (low < high) {
    const mid = Math.ceil((low + high) / 2);
    if (times[mid] <= time) low = mid;
    else high = mid - 1;
  }
  return low;
}

export function sampleTrack(track, time, out = {}) {
  const { times, positions, maxGap } = track;
  const i = lowerSample(times, time);
  const next = Math.min(i + 1, times.length - 1);
  const dt = times[next] - times[i];
  out.visible = time >= times[0] && time <= times.at(-1) && (dt <= maxGap || time === times[i]);
  out.index = i;
  out.alpha = 0;
  out.x = positions[i * 3];
  out.y = positions[i * 3 + 1];
  out.z = positions[i * 3 + 2];
  out.speed = 0;
  out.heading = track.headings[i];
  out.distance = track.distances[i];
  if (dt > 0 && dt <= maxGap) {
    const alpha = Math.max(0, Math.min(1, (time - times[i]) / dt));
    out.alpha = alpha;
    const dx = positions[next * 3] - out.x;
    const dy = positions[next * 3 + 1] - out.y;
    const dz = positions[next * 3 + 2] - out.z;
    out.x += dx * alpha; out.y += dy * alpha; out.z += dz * alpha;
    out.speed = Math.hypot(dx, dy, dz) / dt;
    out.distance += (track.distances[next] - out.distance) * alpha;
  }
  return out;
}

export function advancePlayback(time, delta, start, end, loop) {
  const next = time + delta;
  if (next < end) return { time: Math.max(start, next), ended: false };
  if (!loop) return { time: end, ended: true };
  return { time: start + (next - start) % (end - start), ended: false };
}

export async function fetchText(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`${response.status}: ${url}`);
  const text = await response.text();
  if (text.length > 32 * 1024 * 1024) throw new Error("Файл траекторий больше 32 МБ");
  return text;
}

export async function loadDemo(signal) {
  const base = `${import.meta.env.BASE_URL}demo/project2/`;
  const [object, reportText, droneText, ...cameraTexts] = await Promise.all([
    fetchText(`${base}clean/object_transform_SM_AA_track.csv`, signal),
    fetchText(`${base}clean/report.json`, signal),
    fetchText(`${base}drone-camera5.json`, signal),
    ...[1, 2, 3, 4, 5].map((i) => fetchText(`${base}clean/object_transform_CineCameraActor${i}.csv`, signal)),
  ]);
  const cameras = cameraTexts.map((text, index) => {
    const rows = text.trim().split(/\r?\n/);
    const samples = rows.slice(1).map((row) => {
      const cells = row.split(";");
      return { t: Number(cells[0]) / 30, position: cells.slice(1, 4).map(Number) };
    });
    const rotations = new Float32Array((rows.length - 1) * 3);
    rows.slice(1).forEach((row, i) => rotations.set(row.split(";").slice(4, 7).map(Number), i * 3));
    return {
      ...createTrack({ id: `cam-${index + 1}`, samples, units: "cm" }),
      rotations, focal: Number(rows[0].split(";").at(-1)), sensorWidth: 23.76, sensorHeight: 13.365, name: `Камера ${index + 1}`,
    };
  });
  return createDataset(
    [{ ...parseTrajectoryCsv(object, { id: "truck-1", name: "Cargo Truck · 01" }), kind: "truck", source: "reference" }],
    {
      name: "Project 02", cameras, imageTracks: [parseImageTrack(JSON.parse(droneText))],
      dataQuality: JSON.parse(reportText),
      clips: cameras.map((camera, i) => ({ id: camera.id, name: camera.name, url: `${base}Way1.FinalImage.CineCameraActor${i + 1}.mp4` })),
    },
  );
}
import { parseImageTrack } from "./image-tracks.js";
