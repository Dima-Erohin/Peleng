import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { parseImageTrack, sampleImageTrack } from "./image-tracks.js";

const source = () => JSON.parse(readFileSync(new URL("../../public/demo/project2/drone-camera5.json", import.meta.url), "utf8"));

test("actual drone image track preserves video timing, seed, losses and camera identity", () => {
  const track = parseImageTrack(source());
  assert.equal(track.camera_id, "cam-5");
  assert.equal(track.frame_count, 1001);
  assert.equal(track.width, 1920);
  assert.equal(track.height, 1080);
  assert.equal(sampleImageTrack(track, 7).status, "seed");
  assert.deepEqual(sampleImageTrack(track, 7).bbox, [1292, 185, 86, 55]);
  assert.equal(sampleImageTrack(track, 22).bbox, null);
  assert.equal(sampleImageTrack(track, -1), null);
  assert.equal(sampleImageTrack(track, 40), null);
  assert.equal(track.frames.filter((frame) => frame.bbox !== null).length, track.visible_frames);
  assert.equal(track.visible_frames + track.lost_frames, track.frame_count);
});

test("image tracks reject fake boxes in gaps, invalid timestamps, dimensions and non-finite data", () => {
  for (const mutate of [
    (track) => { track.width = 0; },
    (track) => { track.camera_id = {}; },
    (track) => { track.object_id = 42; },
    (track) => { track.label = {}; },
    (track) => { track.fps = 241; },
    (track) => { track.frames[0].time = 10; },
    (track) => { track.frames[0].bbox = [0, 0, 2000, 10]; },
    (track) => { track.frames[0].correlation = NaN; },
    (track) => { track.frames[0].status = "lost"; },
  ]) {
    const track = source();
    mutate(track);
    assert.throws(() => parseImageTrack(track));
  }
});
