import { createDataset, createTrack, sampleTrack, MAX_TOTAL_SAMPLES, MAX_TRACKS } from "./data.js";

const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const subtract = (a, b) => a.map((value, i) => value - b[i]);
const normalize = (v) => { const length = Math.hypot(...v); return v.map((value) => value / length); };
const MAX_ERROR_PX = 4;
const MIN_ANGLE = Math.PI / 180;

export function cameraAtTime(camera, time, width, height) {
  const pose = sampleTrack(camera, time);
  if (!pose.visible) return null;
  const next = Math.min(pose.index + 1, camera.times.length - 1);
  const angles = [0, 1, 2].map((axis) => {
    const from = camera.rotations[pose.index * 3 + axis] * Math.PI / 180;
    const to = camera.rotations[next * 3 + axis] * Math.PI / 180;
    return from + Math.atan2(Math.sin(to - from), Math.cos(to - from)) * pose.alpha;
  });
  const [roll, pitch, yaw] = angles;
  const [sr, cr, sp, cp, sy, cy] = [Math.sin(roll), Math.cos(roll), Math.sin(pitch), Math.cos(pitch), Math.sin(yaw), Math.cos(yaw)];
  // Unreal: +X forward, +Y right, +Z up; image: +X right, +Y down.
  return {
    origin: [pose.x, pose.y, pose.z],
    forward: [cp * cy, cp * sy, sp],
    right: [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp],
    up: [-cr * sp * cy - sr * sy, -cr * sp * sy + sr * cy, cr * cp],
    fx: width * camera.focal / camera.sensorWidth,
    fy: height * camera.focal / camera.sensorHeight,
    width, height,
  };
}

export function projectPoint(camera, point) {
  const offset = subtract(point, camera.origin), depth = dot(offset, camera.forward);
  if (depth <= 0.05) return null;
  return [camera.width / 2 + camera.fx * dot(offset, camera.right) / depth,
    camera.height / 2 - camera.fy * dot(offset, camera.up) / depth];
}

export function observationRays(imageTracks, cameras, objectId, time) {
  const observations = [];
  for (const track of imageTracks) {
    if (track.object_id !== objectId) continue;
    const frame = track.frames[Math.round(time * track.fps)];
    if (!frame?.bbox || Math.abs(frame.time - time) > 0.51 / track.fps) continue;
    const source = cameras.find((camera) => camera.id === track.camera_id);
    if (!source) throw new Error(`Нет калибровки камеры ${track.camera_id}`);
    const camera = cameraAtTime(source, time, track.width, track.height);
    if (!camera) continue;
    const [x, y, w, h] = frame.bbox, pixel = [x + w / 2, y + h / 2];
    const dx = (pixel[0] - track.width / 2) / camera.fx;
    const dy = (pixel[1] - track.height / 2) / camera.fy;
    const direction = normalize(camera.forward.map((value, i) => value + dx * camera.right[i] - dy * camera.up[i]));
    observations.push({ cameraId: source.id, camera, pixel, origin: camera.origin, direction });
  }
  return observations;
}

function intersectRays(observations) {
  const matrix = Array.from({ length: 3 }, () => [0, 0, 0, 0]);
  for (const { origin, direction } of observations) {
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const a = Number(i === j) - direction[i] * direction[j];
        matrix[i][j] += a;
        matrix[i][3] += a * origin[j];
      }
    }
  }
  for (let i = 0; i < 3; i++) {
    let pivot = i;
    for (let j = i + 1; j < 3; j++) if (Math.abs(matrix[j][i]) > Math.abs(matrix[pivot][i])) pivot = j;
    if (Math.abs(matrix[pivot][i]) < 1e-8) return null;
    [matrix[i], matrix[pivot]] = [matrix[pivot], matrix[i]];
    const divisor = matrix[i][i];
    for (let k = i; k < 4; k++) matrix[i][k] /= divisor;
    for (let j = 0; j < 3; j++) {
      if (j === i) continue;
      const factor = matrix[j][i];
      for (let k = i; k < 4; k++) matrix[j][k] -= factor * matrix[i][k];
    }
  }
  const point = matrix.map((row) => row[3]);
  return point.every(Number.isFinite) ? point : null;
}

function errorPx(observation, point) {
  const projected = projectPoint(observation.camera, point);
  return projected ? Math.hypot(...subtract(projected, observation.pixel)) : Infinity;
}

export function triangulateObservations(observations) {
  if (observations.length < 2) return { status: "insufficient", position: null, used: [] };
  let best = null;
  for (let i = 0; i < observations.length; i++) {
    for (let j = i + 1; j < observations.length; j++) {
      const angle = Math.acos(Math.min(1, Math.abs(dot(observations[i].direction, observations[j].direction))));
      if (angle < MIN_ANGLE) continue;
      const position = intersectRays([observations[i], observations[j]]);
      if (!position) continue;
      const used = observations.filter((view) => errorPx(view, position) <= MAX_ERROR_PX);
      if (used.length < 2) continue;
      const refined = intersectRays(used);
      if (!refined) continue;
      const errors = used.map((view) => errorPx(view, refined));
      if (Math.max(...errors) > MAX_ERROR_PX) continue;
      const rmse = Math.sqrt(errors.reduce((sum, error) => sum + error * error, 0) / errors.length);
      if (!best || used.length > best.used.length || (used.length === best.used.length && rmse < best.reprojectionPx)) {
        best = { status: "triangulated", position: refined, used: used.map((view) => view.cameraId), reprojectionPx: rmse, angleDeg: angle * 180 / Math.PI };
      }
    }
  }
  return best || { status: "rejected", position: null, used: [] };
}

export function reconstructDataset(dataset, imageTracks = dataset.imageTracks) {
  if (imageTracks.length > MAX_TRACKS) throw new Error(`Максимум ${MAX_TRACKS} треков наблюдений`);
  if (imageTracks.reduce((sum, track) => sum + track.frame_count, 0) > MAX_TOTAL_SAMPLES) {
    throw new Error(`Максимум ${MAX_TOTAL_SAMPLES} наблюдений камер`);
  }
  const keys = new Set();
  for (const track of imageTracks) {
    const key = JSON.stringify([track.camera_id, track.object_id]);
    if (keys.has(key)) throw new Error(`Дублируется объект ${track.object_id} в камере ${track.camera_id}`);
    keys.add(key);
    const camera = dataset.cameras.find((item) => item.id === track.camera_id);
    if (!camera || ![camera.focal, camera.sensorWidth, camera.sensorHeight].every((n) => Number.isFinite(n) && n > 0) ||
        camera.rotations?.length !== camera.times.length * 3 || !camera.rotations.every(Number.isFinite)) {
      throw new Error(`Для камеры ${track.camera_id} нужны положения, вращения, фокусное расстояние и размер сенсора`);
    }
  }
  const reconstructions = [], tracks = dataset.tracks.filter((track) => track.source !== "triangulated");
  for (const id of new Set(imageTracks.map((track) => track.object_id))) {
    if (tracks.some((track) => track.id === id)) throw new Error(`ID наблюдения ${id} уже занят траекторией`);
    const views = imageTracks.filter((track) => track.object_id === id);
    const fps = dataset.fps, count = Math.floor(Math.max(...views.map((view) => (view.frame_count - 1) / view.fps)) * fps + 0.0001) + 1;
    if (count > 200_000) throw new Error("Слишком длинная последовательность наблюдений");
    const frames = Array.from({ length: count }, (_, frame) => {
      const time = frame / fps;
      const observations = observationRays(views, dataset.cameras, id, time);
      return { time, cameras: observations.map((view) => view.cameraId), ...triangulateObservations(observations) };
    });
    // A three-frame symmetric filter never fills missing detections. Reprojection
    // bounds keep smoothing from moving points away from their image evidence.
    const smoothed = frames.map((frame, i) => {
      if (!frame.position || !frames[i - 1]?.position || !frames[i + 1]?.position) return frame;
      const position = frame.position.map((value, a) => (frames[i - 1].position[a] + 2 * value + frames[i + 1].position[a]) / 4);
      const used = observationRays(views, dataset.cameras, id, frame.time).filter((view) => frame.used.includes(view.cameraId));
      const errors = used.map((view) => errorPx(view, position));
      if (Math.max(...errors) > MAX_ERROR_PX) return frame;
      return { ...frame, position, rawPosition: frame.position, reprojectionPx: Math.sqrt(errors.reduce((sum, error) => sum + error * error, 0) / errors.length) };
    });
    const valid = smoothed.filter((frame) => frame.position);
    const name = views[0].label === "drone" ? `Дрон · ${id === "drone-1" ? "01" : id}` : id;
    const kind = views[0].label === "drone" ? "drone" : "object";
    if (valid.length >= 2) {
      tracks.push({ ...createTrack({ id, name, units: "m", maxGap: 1.5 / fps,
        samples: valid.map((frame) => ({ t: frame.time, position: frame.position })) }), kind, source: "triangulated" });
    }
    reconstructions.push({ id, name, kind, frames: smoothed, fps, validCount: valid.length,
      cameraIds: views.map((view) => view.camera_id), hasTrack: valid.length >= 2 });
  }
  const result = createDataset(tracks, { ...dataset, imageTracks });
  const start = imageTracks.length ? Math.min(0, result.start) : result.start;
  const end = Math.max(result.end, ...imageTracks.map((view) => (view.frame_count - 1) / view.fps));
  return { ...result, start, end, duration: end - start, reconstructions, truckObservations: dataset.truckObservations };
}
