import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createTrack, parseTrajectoryCsv, sampleTrack } from "./data.js";
import { smoothTruckTrack } from "./truck-motion.js";

test("presentation filter preserves constant velocity and the raw measurements", () => {
  const track = createTrack({ id: "truck", units: "m", samples: Array.from({ length: 60 }, (_, i) => ({
    t: i / 30, position: [10 + i / 2, i / 3, 3],
  })) });
  const raw = [...track.positions];
  const display = smoothTruckTrack(track);
  assert.notEqual(display.positions, track.positions);
  assert.deepEqual([...track.positions], raw);
  display.positions.forEach((value, index) => assert.ok(Math.abs(value - raw[index]) < 1e-8));
});

test("isolated occlusion spike is suppressed without filling missing intervals", () => {
  const samples = Array.from({ length: 90 }, (_, i) => ({ t: i / 30, position: [i / 2, 0, 3] }));
  samples[40].position[1] = 5;
  const track = createTrack({ id: "truck", units: "m", samples });
  const display = smoothTruckTrack(track);
  assert.ok(Math.abs(display.positions[40 * 3 + 1]) < 0.2);
  const gapped = smoothTruckTrack(createTrack({ id: "gap", units: "m", samples: [
    { t: 0, position: [0, 0, 0] }, { t: 0.1, position: [1, 0, 0] },
    { t: 2, position: [100, 0, 0] }, { t: 2.1, position: [101, 0, 0] },
  ] }));
  assert.equal(sampleTrack(gapped, 1).visible, false);
  assert.ok(Math.abs(gapped.positions[3] - 1) < 1e-8);
});

test("regenerated estimate respects the position-step bound and display has no abrupt heading flips", () => {
  const raw = parseTrajectoryCsv(readFileSync(new URL("../../public/demo/project2/estimate/truck.csv", import.meta.url), "utf8"));
  const display = smoothTruckTrack(raw);
  const maxStep = (track) => Math.max(...Array.from({ length: track.times.length - 1 }, (_, i) =>
    Math.hypot(...[0, 1, 2].map((axis) => track.positions[(i + 1) * 3 + axis] - track.positions[i * 3 + axis]))));
  assert.ok(maxStep(raw) <= 2.00001, maxStep(raw));
  assert.ok(maxStep(display) < 1.8, maxStep(display));
  assert.ok(maxStep(display) < maxStep(raw));
  for (let i = 1; i < display.times.length; i++) {
    const turn = Math.abs(Math.atan2(Math.sin(display.headings[i] - display.headings[i - 1]),
      Math.cos(display.headings[i] - display.headings[i - 1])));
    assert.ok(turn <= Math.PI / 2 * (display.times[i] - display.times[i - 1]) + 1e-10);
  }
});
