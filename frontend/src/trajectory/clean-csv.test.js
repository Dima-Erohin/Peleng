import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { cleanCsv } from "../../scripts/prepare-demo.mjs";
import { parseTrajectoryCsv } from "./data.js";

const base = new URL("../../public/demo/project2/", import.meta.url);
const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");

test("clean demo has valid UTF-8, all frames, intact originals and reproducible estimates", () => {
  const report = JSON.parse(readFileSync(new URL("clean/report.json", base), "utf8"));
  assert.equal(report.originalsPreserved, true);
  let repaired = 0;
  for (const file of report.files) {
    const original = readFileSync(new URL(file.name, base));
    const cleaned = readFileSync(new URL(`clean/${file.name}`, base));
    const text = new TextDecoder("utf-8", { fatal: true }).decode(cleaned);
    assert.equal(digest(original), file.originalSha256);
    assert.equal(digest(cleaned), file.cleanSha256);
    const regenerated = cleanCsv(new TextDecoder("windows-1251").decode(original), file.name);
    assert.equal(regenerated.csv, text);
    assert.deepEqual(regenerated.repairs, file.repairs);
    assert.equal(regenerated.frameCount, 1002);
    repaired += file.repairs.length;
    if (file.name.includes("SM_AA")) {
      const track = parseTrajectoryCsv(text);
      assert.equal(track.times.length, 1002);
      assert.deepEqual(track.skippedFrames, []);
      assert.deepEqual(file.repairs.map((entry) => [entry.frame, entry.column, entry.estimated]),
        [[402, "Loc_X", 8.80835], [691, "Loc_Y", 7.87305]]);
    }
    const oldRows = new TextDecoder("windows-1251").decode(original).trim().split(/\r?\n/);
    const newRows = text.trim().split(/\r?\n/);
    oldRows.forEach((row, i) => row.split(";").forEach((value, j) => {
      const key = oldRows[0].split(";")[j];
      if (!file.repairs.some((repair) => repair.frame === i - 1 && repair.column === key)) {
        assert.equal(newRows[i].split(";")[j], value);
      }
    }));
  }
  assert.equal(repaired, 2);
});

test("repair refuses consecutive gaps, endpoints, bad frame sequence and rotation errors", () => {
  const head = "Frame;Loc_X;Loc_Y;Loc_Z\n";
  for (const rows of [
    "0;bad;2;3\n1;1;2;3",
    "0;1;2;3\n1;bad;2;3",
    "0;1;2;3\n1;bad;2;3\n2;bad;2;3\n3;4;2;3",
    "0;1;2;3\n2;2;2;3",
  ]) assert.throws(() => cleanCsv(head + rows, "test"));
  assert.throws(() => cleanCsv("Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw\n0;0;0;0;0;0;bad\n1;0;0;0;0;0;0", "camera"));
});
