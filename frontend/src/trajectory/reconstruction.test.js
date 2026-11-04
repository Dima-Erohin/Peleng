import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createDataset, createTrack, sampleTrack } from "./data.js";
import { parseImageTrack } from "./image-tracks.js";
import { cameraAtTime, observationRays, projectPoint, reconstructDataset, triangulateObservations } from "./reconstruction.js";

function camera(id, origin) {
  return { ...createTrack({ id, units: "m", maxGap: 2, samples: [{ t: 0, position: origin }, { t: 1, position: origin }] }),
    rotations: new Float32Array([0, 20, 0, 0, 20, 0]), focal: 20, sensorWidth: 36, sensorHeight: 20.25 };
}
const cameras = [camera("cam-1", [0, -5, 0]), camera("cam-2", [0, 5, 0]), camera("cam-3", [0, -10, 4]), camera("cam-4", [0, 10, 2])];
const pointAt = (time) => [30 + time, 0, 20 + time / 2];
function imageTrack(camera, { lost = [], shift = [0, 0], count = 31, fps = 30 } = {}) {
  return parseImageTrack({ version: 1, space: "image", camera_id: camera.id, object_id: "drone-1",
    label: "drone", width: 1920, height: 1080, fps, frame_count: count,
    frames: Array.from({ length: count }, (_, frame) => {
      const time = frame / fps;
      const pixel = projectPoint(cameraAtTime(camera, time, 1920, 1080), pointAt(time));
      return { frame, time, bbox: lost.includes(frame) ? null : [pixel[0] - 4 + shift[0], pixel[1] - 4 + shift[1], 8, 8],
        correlation: 1, status: lost.includes(frame) ? "lost" : "tracked" };
    }) });
}
function base(views, selectedCameras = cameras) {
  const truck = createTrack({ id: "truck-1", units: "m", samples: [{ t: 0, position: [0, 0, 0] }, { t: 1, position: [1, 0, 0] }] });
  return createDataset([truck], { cameras: selectedCameras, imageTracks: views });
}

test("calibrated rays recover airborne XYZ and bounded reprojection, without a floor projection", () => {
  const views = cameras.slice(0, 2).map((camera) => imageTrack(camera));
  const rays = observationRays(views, cameras, "drone-1", 0.5);
  const result = triangulateObservations(rays);
  assert.equal(result.status, "triangulated");
  assert.ok(Math.hypot(...result.position.map((x, i) => x - pointAt(0.5)[i])) < 1e-8);
  assert.ok(result.position[2] > 20);
  assert.ok(result.reprojectionPx < 1e-8);
  const dataset = reconstructDataset(base(views));
  assert.equal(dataset.reconstructions[0].validCount, 31);
  assert.equal(dataset.tracks.find((track) => track.id === "drone-1").kind, "drone");
  assert.ok(dataset.reconstructions[0].frames[15].rawPosition);
});

test("one camera and camera rotation do not invent depth, and lost detections have no rays", () => {
  const view = imageTrack(cameras[0], { lost: [10] });
  const dataset = reconstructDataset(base([view]));
  assert.equal(dataset.reconstructions[0].validCount, 0);
  assert.equal(dataset.tracks.length, 1);
  assert.equal(dataset.reconstructions[0].frames[0].status, "insufficient");
  assert.equal(dataset.reconstructions[0].frames[10].cameras.length, 0);
  assert.equal(observationRays([view], cameras, "drone-1", 10 / 30).length, 0);
});

test("outlying views are rejected with consensus and no interpolation bridges missing measurements", () => {
  const views = cameras.map((camera, i) => imageTrack(camera, { shift: i === 3 ? [250, 100] : [0, 0], lost: [10] }));
  const dataset = reconstructDataset(base(views));
  assert.deepEqual(dataset.reconstructions[0].frames[5].used, ["cam-1", "cam-2", "cam-3"]);
  const track = dataset.tracks.find((track) => track.id === "drone-1");
  assert.equal(dataset.reconstructions[0].frames[10].position, null);
  assert.equal(sampleTrack(track, 10 / 30).visible, false);
  assert.equal(dataset.reconstructions[0].validCount, 30);
});

test("parallel or backwards rays and inconsistent observations cannot produce a 3D location", () => {
  const views = cameras.slice(0, 2).map((camera) => imageTrack(camera));
  const rays = observationRays(views, cameras, "drone-1", 0);
  assert.equal(triangulateObservations([rays[0], { ...rays[1], direction: rays[0].direction }]).position, null);
  assert.equal(triangulateObservations(rays.map((ray) => ({ ...ray, direction: ray.direction.map((n) => -n),
    camera: { ...ray.camera, forward: ray.camera.forward.map((n) => -n) } }))).position, null);
  assert.equal(triangulateObservations([rays[0], { ...rays[1], pixel: [2000, 2000] }]).position, null);
});

test("duplicate camera/object pairs, absent calibration and collisions with reference IDs are explicit errors", () => {
  const view = imageTrack(cameras[0]);
  assert.throws(() => reconstructDataset(base([view, view])), /Дублируется/);
  assert.throws(() => reconstructDataset(base([view], [])), /камеры/);
  assert.throws(() => reconstructDataset(base([view], [{ ...cameras[0], sensorWidth: 0 }])), /размер сенсора/);
  assert.throws(() => reconstructDataset(base([{ ...view, object_id: "truck-1" }])), /уже занят/);
  const twice = reconstructDataset(reconstructDataset(base(cameras.slice(0, 2).map((camera) => imageTrack(camera)))));
  assert.equal(twice.tracks.length, 2);
});

test("camera pose respects Unreal roll, pitch and yaw and preserves physical focal/sensor geometry", () => {
  const cam = { ...camera("rotated", [4, 5, 6]), rotations: new Float32Array([10, 30, 70, 10, 30, 70]) };
  const pose = cameraAtTime(cam, 0, 1920, 1080);
  for (const axis of ["forward", "right", "up"]) assert.ok(Math.abs(Math.hypot(...pose[axis]) - 1) < 1e-12);
  const inFront = pose.origin.map((x, i) => x + pose.forward[i] * 100);
  assert.deepEqual(projectPoint(pose, inFront).map(Math.round), [960, 540]);
  assert.equal(projectPoint(pose, pose.origin.map((x, i) => x - pose.forward[i] * 100)), null);
});

test("actual Project 02 drone reports no XYZ instead of silently turning a 2D fixture into ground truth", () => {
  const text = readFileSync(new URL("../../public/demo/project2/clean/object_transform_CineCameraActor5.csv", import.meta.url), "utf8").trim().split(/\r?\n/);
  const rows = text.slice(1).map((row) => row.split(";").slice(0, 7).map(Number));
  const cam = { ...createTrack({ id: "cam-5", units: "cm", samples: rows.map((row) => ({ t: row[0] / 30, position: row.slice(1, 4) })) }),
    rotations: new Float32Array(rows.flatMap((row) => row.slice(4))), focal: 20, sensorWidth: 23.76, sensorHeight: 13.365 };
  const view = parseImageTrack(JSON.parse(readFileSync(new URL("../../public/demo/project2/drone-camera5.json", import.meta.url), "utf8")));
  const dataset = reconstructDataset(base([view], [cam]));
  assert.equal(dataset.reconstructions[0].validCount, 0);
  assert.equal(dataset.reconstructions[0].frames.length, 1001);
  const ray = observationRays([view], [cam], "drone-1", 7)[0];
  assert.deepEqual(ray.pixel, [1335, 212.5]);
  assert.ok(ray.direction[2] > 0);
});

test("smoothing reduces known synthetic jitter without exceeding image evidence gates", () => {
  const views = cameras.slice(0, 3).map((camera) => imageTrack(camera));
  for (const view of views) {
    view.frames.forEach((frame, i) => { frame.bbox[0] += i % 2 ? 0.7 : -0.7; });
  }
  const frames = reconstructDataset(base(views)).reconstructions[0].frames.slice(1, -1);
  const error = (position, time) => Math.hypot(...position.map((x, i) => x - pointAt(time)[i]));
  const rawError = frames.reduce((sum, frame) => sum + error(frame.rawPosition || frame.position, frame.time), 0);
  const filteredError = frames.reduce((sum, frame) => sum + error(frame.position, frame.time), 0);
  assert.ok(filteredError < rawError / 2);
  assert.ok(frames.every((frame) => frame.reprojectionPx <= 4));
});

test("a rotated camera at the same origin has no stereo baseline", () => {
  const a = camera("a", [0, 0, 0]);
  const b = { ...camera("b", [0, 0, 0]), rotations: new Float32Array([5, 30, 5, 5, 30, 5]) };
  const dataset = reconstructDataset(base([imageTrack(a), imageTrack(b)], [a, b]));
  assert.equal(dataset.reconstructions[0].validCount, 0);
});

test("an isolated accepted XYZ is retained without manufacturing a trajectory or bridging gaps", () => {
  const views = cameras.slice(0, 2).map((camera) => imageTrack(camera, { lost: Array.from({ length: 31 }, (_, i) => i).filter((i) => i !== 15) }));
  const dataset = reconstructDataset(base(views));
  assert.equal(dataset.reconstructions[0].validCount, 1);
  assert.equal(dataset.reconstructions[0].hasTrack, false);
  assert.ok(dataset.reconstructions[0].frames[15].position);
  assert.equal(dataset.tracks.length, 1);
});
