import test from "node:test";
import assert from "node:assert/strict";
import { createTrack } from "./data.js";
import { overviewBounds, routeGeometry } from "./minimap-scene.js";
import { minimapStyle } from "./minimap-style.js";

test("overlapping reference and estimate remain distinct by color, width, dash and draw order", () => {
  const reference = minimapStyle({ source: "reference" });
  const estimate = minimapStyle({ source: "estimate" });
  assert.notEqual(reference.color, estimate.color);
  assert.equal(reference.dashed, false);
  assert.equal(estimate.dashed, true);
  assert.ok(reference.width > estimate.width);
  assert.ok(estimate.order > reference.order + 1);
  assert.deepEqual(minimapStyle({ live: true }), estimate);
});

test("3D overview includes height and cameras in its bounds", () => {
  const track = createTrack({ id: "truck", units: "m", samples: [
    { t: 0, position: [-10, 5, 3] }, { t: 1, position: [20, 15, 8] },
  ] });
  const camera = createTrack({ id: "camera", units: "m", samples: [
    { t: 0, position: [100, -30, 50] }, { t: 1, position: [100, -30, 50] },
  ] });
  const bounds = overviewBounds([track, camera]);
  assert.deepEqual(bounds.min.toArray(), [-10, -30, 3]);
  assert.deepEqual(bounds.max.toArray(), [100, 15, 50]);
  assert.equal(overviewBounds([]).isEmpty(), false);
});

test("3D overview preserves actual XYZ and does not connect tracking gaps", () => {
  const track = createTrack({ id: "truck", units: "m", samples: [
    { t: 0, position: [0, 0, 0] }, { t: 0.1, position: [1, 2, 3] },
    { t: 2, position: [10, 20, 30] }, { t: 2.1, position: [11, 22, 33] },
  ] });
  const { geometry, ends } = routeGeometry(track);
  assert.deepEqual([...geometry.attributes.position.array], [0, 0, 0, 1, 2, 3, 10, 20, 30, 11, 22, 33]);
  assert.deepEqual(ends, [0, 2, 2, 4]);
  geometry.dispose();
});
