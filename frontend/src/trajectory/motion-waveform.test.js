import test from "node:test";
import assert from "node:assert/strict";
import { createTrack } from "./data.js";
import { motionScale, motionWaveform } from "./motion-waveform.js";

const track = (samples) => createTrack({ id: "test", units: "m", samples: samples.map(([t, x]) => ({ t, position: [x, 0, 0] })) });

test("waveform follows actual speed, stays flat at rest and does not change while paused", () => {
  const source = track([[0, 0], [.5, 0], [1, 1], [1.5, 3], [2, 3], [2.5, 3]]);
  const scale = motionScale(source);
  assert.equal(scale, 4);
  const wave = motionWaveform(source, 2, scale, 5, 2);
  assert.deepEqual(wave, [0, .5, 1, 0, 0]);
  assert.deepEqual(motionWaveform(source, 2, scale, 5, 2), wave);
  assert.ok(motionWaveform(source, .25, scale).slice(0, -3).every((value) => value === null));
});

test("missing XYZ and time gaps never become invented motion", () => {
  assert.ok(motionWaveform(null, 10, 1).every((value) => value === null));
  const source = track([[0, 0], [.5, 1], [2, 100], [2.5, 100]]);
  assert.equal(motionScale(source), 2);
  assert.equal(motionWaveform(source, 1, 2, 2, .25)[1], null);
  const stopped = track([[0, 0], [.5, 0]]);
  assert.equal(motionScale(stopped), .01);
  assert.ok(motionWaveform(stopped, .5, .01, 4, .5).every((value) => value === 0));
});
