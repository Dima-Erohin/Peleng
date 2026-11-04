import test from "node:test";
import assert from "node:assert/strict";
import { buildOnlineTrack, onlineClock, onlineStats } from "./online.js";

const point = (frame, pos, extra = {}) => ({ type: "point", frame, t: frame / 30, pos, lag_ms: 500, proc_ms: 400, ...extra });

test("online track needs two located points and converts centimetres to metres", () => {
  assert.equal(buildOnlineTrack([point(0, null), point(10, [100, 200, 300])]), null);
  const track = buildOnlineTrack([point(0, null), point(10, [100, 200, 300]), point(40, [400, 200, 300], { vel: [300, 0, 0] })]);
  assert.equal(track.times.length, 2);
  assert.deepEqual([...track.positions.slice(3, 6)], [4, 2, 3]);
  assert.deepEqual(track.liveVel, [3, 0, 0]);
  assert.equal(track.live, true);
  // шаг обработки ~1 с не должен рвать след
  assert.ok(track.maxGap >= 1);
});

test("online clock follows server time", () => {
  const clock = onlineClock({ server_time: 1000, t0: 990 }, 2000);
  const value = clock();
  // браузерные часы на 1000 с впереди серверных: время прогона ≈ (now - 1000) - 990 + ...
  assert.ok(Number.isFinite(value));
  const later = onlineClock({ server_time: Date.now() / 1000, t0: Date.now() / 1000 - 5 })();
  assert.ok(Math.abs(later - 5) < 0.5);
});

test("online stats report rate and medians", () => {
  const stats = onlineStats([point(0, [0, 0, 0]), point(30, [1, 0, 0]), point(60, [2, 0, 0])]);
  assert.equal(stats.count, 3);
  assert.ok(Math.abs(stats.rate - 1) < 1e-9);
  assert.equal(stats.lagMs, 500);
});
