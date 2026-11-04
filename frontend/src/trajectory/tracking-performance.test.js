import test from "node:test";
import assert from "node:assert/strict";
import { createTrack } from "./data.js";
import { metricPath, trackingPerformance } from "./tracking-performance.js";

const reference = createTrack({ id: "reference", units: "m", maxGap: 2,
  samples: [0, 1, 2].map((t) => ({ t, position: [t, 0, 0] })) });

test("performance compares raw XYZ with synchronized reference and converts live centimetres", () => {
  const points = [0, 1, 2].map((t) => ({ t, mode: "tri", pos: [t * 100, 300, 400] }));
  const rows = trackingPerformance({ reference, points, live: true });
  assert.deepEqual(rows.map((p) => p.error), [5, 5, 5]);
  assert.deepEqual(rows.map((p) => p.hz), [null, 1, 1]);
  const estimate = createTrack({ id: "estimate", units: "m", maxGap: 2,
    samples: [0, 1, 2].map((t) => ({ t, position: [t, 3, 4] })) });
  const offline = trackingPerformance({ reference, estimate, points });
  assert.deepEqual(offline.map((p) => p.error), [5, 5, 5]);
  assert.ok(offline.every((p) => p.hz === null));
});

test("frequency uses latest 10 processed timestamps, not inverse computation duration", () => {
  const points = Array.from({ length: 15 }, (_, i) => ({ t: i * 0.1, mode: "tri", proc_ms: 20 }));
  const rows = trackingPerformance({ points, live: true });
  assert.ok(Math.abs(rows.at(-1).hz - 10) < 1e-9);
  assert.equal(rows.at(-1).error, null);
});

test("recorded error uses the current estimate timestamps without historical observation metadata", () => {
  const estimate = createTrack({ id: "estimate", units: "m", maxGap: 2,
    samples: [0, 1, 2].map((t) => ({ t, position: [t, 3, 4] })) });
  const rows = trackingPerformance({ reference, estimate });
  assert.deepEqual(rows.map(({ t, error, hz }) => [t, error, hz]), [[0, 5, null], [1, 5, null], [2, 5, null]]);
  assert.deepEqual(trackingPerformance({ reference }), []);
});

test("missing XYZ and reference gaps never become zero error or connected lines", () => {
  const points = [{ t: 0, mode: "none", pos: [0, 0, 0] }, { t: 1, mode: "tri" },
    { t: 3, mode: "tri", pos: [0, 0, 0] }];
  assert.ok(trackingPerformance({ reference, points, live: true }).every((p) => p.error === null));
  const gap = createTrack({ id: "gap", units: "m", samples: [{ t: 0, position: [0, 0, 0] }, { t: 2, position: [0, 0, 0] }] });
  assert.equal(trackingPerformance({ reference: gap, points: [{ t: 1, mode: "tri", pos: [0, 0, 0] }], live: true })[0].error, null);
  const path = metricPath([{ t: 0, error: 1 }, { t: 0.5, error: null }, { t: 1, error: 2 }, { t: 3, error: 1 }], "error", 0, 3, 5);
  assert.equal((path.match(/M/g) || []).length, 3);
  assert.ok(!path.includes("NaN"));
});
