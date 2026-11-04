import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { advancePlayback, createDataset, createTrack, lowerSample, parseTrajectoryCsv, sampleTrack } from "./data.js";

const csv = "Frame;Loc_X;Loc_Y;Loc_Z\n0;0;0;300\n15;100;200;400\n30;200;400;500";

test("CSV converts centimeters, preserves Z and interpolates by timestamps", () => {
  const track = parseTrajectoryCsv(csv);
  assert.equal(track.times.at(-1), 1);
  const p = sampleTrack(track, 0.25);
  assert.deepEqual([p.x, p.y, p.z], [0.5, 1, 3.5]);
  assert.equal(p.visible, true);
  assert.ok(Math.abs(p.speed - Math.sqrt(24)) < 1e-6);
  assert.ok(Math.abs(p.heading - Math.atan2(2, 1)) < 1e-6);
});

test("rejects missing data, duplicate/out-of-order frames, invalid fps and camera CSV", () => {
  for (const text of [
    "Frame;Loc_X;Loc_Y;Loc_Z\n0;;2;3\n1;1;2;3",
    csv.replace("15;100", "0;100"), csv.replace("15;100", "-1;100"),
    csv.replace("100;200", "Infinity;200"), "x;y\n1;2",
    csv.replace("Loc_Z", "Loc_Z;Rot_Yaw"),
  ]) assert.throws(() => parseTrajectoryCsv(text));
  assert.throws(() => parseTrajectoryCsv(csv, { fps: 0 }));
  assert.throws(() => parseTrajectoryCsv(csv, { units: "feet" }));
});

test("missing detections are gaps, not invented long-distance motion", () => {
  const track = parseTrajectoryCsv(csv.replace("15;100", "29;100"));
  assert.equal(sampleTrack(track, 0).visible, true);
  assert.equal(sampleTrack(track, 0.5).visible, false);
  assert.equal(sampleTrack(track, -1).visible, false);
  assert.equal(sampleTrack(track, 2).visible, false);
  assert.equal(sampleTrack(track, 1).visible, true);
  assert.equal(track.distances[1], 0);
});

test("playback is independent of frame rate, loops with remainder, ends exactly", () => {
  let time = 0;
  for (let i = 0; i < 144; i++) time = advancePlayback(time, 1 / 144, 0, 10, false).time;
  assert.ok(Math.abs(time - 1) < 1e-10);
  assert.deepEqual(advancePlayback(9, 3, 0, 10, true), { time: 2, ended: false });
  assert.deepEqual(advancePlayback(9, 3, 0, 10, false), { time: 10, ended: true });
});

test("binary lookup and multi-object dataset use global timestamps", () => {
  const first = parseTrajectoryCsv(csv, { id: "a" });
  const second = createTrack({ id: "b", units: "m", samples: [{ t: 4, position: [0, 0, 0] }, { t: 5, position: [1, 2, 3] }] });
  const dataset = createDataset([first, second]);
  assert.equal(dataset.duration, 5);
  assert.equal(lowerSample(first.times, 0.75), 1);
  assert.equal(lowerSample(first.times, 100), 2);
  assert.throws(() => createDataset([first, first]));
  assert.notEqual(dataset.tracks[0].color, dataset.tracks[1].color);
  assert.throws(() => createDataset([first], { fps: 0 }));
  assert.throws(() => createTrack({ id: "", units: "m", samples: [] }));
});

test("keeps measurement precision and stable heading when an object stops", () => {
  const track = createTrack({
    id: "precise", units: "m",
    samples: [
      { t: 0, position: [10000.000123, 2, 3] },
      { t: 0.1, position: [10000.000123, 3, 3] },
      { t: 0.2, position: [10000.000123, 3, 3] },
      ...Array.from({ length: 100 }, (_, i) => ({ t: (i + 3) / 10, position: [10000.000123, 3, 3] })),
    ],
  });
  const sample = sampleTrack(track, 10.2);
  assert.equal(sample.x, 10000.000123);
  assert.equal(sample.heading, Math.PI / 2);
  assert.equal(sample.speed, 0);
});

test("supplied fixture reports the two corrupt samples and keeps global timing at 30 fps", () => {
  const text = readFileSync(new URL("../../public/demo/project2/object_transform_SM_AA_track.csv", import.meta.url), "utf8");
  assert.throws(() => parseTrajectoryCsv(text), /Строка 404/);
  const track = parseTrajectoryCsv(text, { skipInvalidPositions: true });
  assert.equal(track.times.length, 1000);
  assert.deepEqual(track.skippedFrames, [402, 691]);
  assert.equal(track.times.at(-1), 1001 / 30);
  const p = sampleTrack(track, 0);
  assert.deepEqual([p.x, p.y, p.z], [-215, -128, 3]);
  assert.ok(track.distances.at(-1) > 400);
});
