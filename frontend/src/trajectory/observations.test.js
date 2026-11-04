import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { observationAt, trackingQuality, frameBoxes, parseTruckObservations, observationsMatchTrajectory } from "./observations.js";
import { createHash } from "node:crypto";

test("reliability metadata is never attributed to a different trajectory export", async () => {
  const csv = "Frame;Loc_X;Loc_Y;Loc_Z\n0;1;2;3\n";
  const observations = { trajectory_sha256: createHash("sha256").update(csv).digest("hex") };
  assert.equal(await observationsMatchTrajectory(observations, csv), true);
  assert.equal(await observationsMatchTrajectory(observations, csv.replace(/\n/g, "\r\n")), true);
  assert.equal(await observationsMatchTrajectory(observations, `${csv}1;4;5;6\n`), false);
  assert.equal(await observationsMatchTrajectory({}, csv), false);
});

test("exported demo observations validate and preserve camera/detection indices", () => {
  const raw = JSON.parse(readFileSync(new URL("../../public/demo/project2/estimate/truck_observations.json", import.meta.url), "utf8"));
  const value = parseTruckObservations(raw);
  assert.equal(value.points.length, 1001);
  assert.equal(trackingQuality(value.points[0], 0).level, "lost");
  assert.ok(value.points.some((p) => p.mode === "mono"));
  assert.ok(value.points.some((p) => p.mode === "predict"));
  assert.ok(value.points.some((p) => p.limited_update));
  assert.throws(() => parseTruckObservations({ ...raw, fps: 0 }), /неверный формат/);
  assert.throws(() => parseTruckObservations({ ...raw, sizes: [] }), /неверный формат/);
  assert.throws(() => parseTruckObservations({ ...raw, points: [raw.points[1], raw.points[0]] }), /неверный формат/);
});

test("video observations never select a future detection or hold an old box over a newer frame", () => {
  const point = { frame: 30, t: 1, mode: "tri", n_cams: 2, dets: [[[10, 20, 50, 60, 0.8, "pose"]]] };
  assert.equal(observationAt([point], 0.9), null);
  assert.equal(observationAt([point], 1), point);
  assert.equal(frameBoxes(point, 0, 1, 30).length, 1);
  assert.equal(frameBoxes(point, 0, 1.1, 30).length, 0);
  assert.equal(frameBoxes(point, 0, 0.9, 30).length, 0);
  assert.equal(frameBoxes(point, 1, 1, 30).length, 0);
});

test("tracking quality distinguishes measurements, depth ambiguity, prediction, corrections and stale data", () => {
  const p = { t: 1, mode: "tri", n_cams: 2 };
  assert.equal(trackingQuality(p, 1).level, "measured");
  assert.equal(trackingQuality({ ...p, n_cams: 1, mode: "mono" }, 1).label, "Один ракурс");
  assert.equal(trackingQuality({ ...p, mode: "predict" }, 1).label, "Прогноз");
  assert.equal(trackingQuality({ ...p, limited_update: true }, 1).label, "Восстановление трека");
  assert.equal(trackingQuality(p, 3).label, "Данные устарели");
  assert.equal(trackingQuality({ ...p, mode: "none" }, 1).level, "lost");
  assert.equal(trackingQuality(null, 1).level, "lost");
});
