import { createHash } from "node:crypto";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const root = new URL("../public/demo/project2/", import.meta.url);
const files = [
  "object_transform_SM_AA_track.csv",
  ...[1, 2, 3, 4, 5].map((i) => `object_transform_CineCameraActor${i}.csv`),
];

export function cleanCsv(text, name) {
  const rows = text.replace(/^\uFEFF/, "").trim().split(/\r?\n/).map((row) => row.split(";"));
  const header = rows[0];
  const required = ["Frame", "Loc_X", "Loc_Y", "Loc_Z", ...(header.includes("Rot_Yaw") ? ["Rot_Roll", "Rot_Pitch", "Rot_Yaw"] : [])];
  const columns = required.map((key) => header.indexOf(key));
  if (columns.some((index) => index < 0)) throw new Error(`${name}: missing required columns`);
  const finite = (value) => value != null && value.trim() !== "" && Number.isFinite(Number(value));
  const repairs = [];
  for (let i = 1; i < rows.length; i++) {
    const frame = Number(rows[i][columns[0]]);
    if (!finite(rows[i][columns[0]]) || !Number.isInteger(frame) || frame < 0 ||
      (i > 1 && frame !== Number(rows[i - 1][columns[0]]) + 1)) {
      throw new Error(`${name}: non-consecutive frame at line ${i + 1}`);
    }
    for (const column of columns.slice(1)) {
      if (finite(rows[i][column])) continue;
      // Only a single missing position bounded by measured neighbors can be
      // estimated. Never fill endpoints, rotations, or runs of invalid samples.
      if (!header[column].startsWith("Loc_") || i <= 1 || i >= rows.length - 1 ||
          !finite(rows[i - 1][column]) || !finite(rows[i + 1][column]) ||
          repairs.some((repair) => repair.frame === frame - 1 && repair.column === header[column])) {
        throw new Error(`${name}: cannot estimate ${header[column]} at frame ${frame}`);
      }
      const before = Number(rows[i - 1][column]), after = Number(rows[i + 1][column]);
      const value = Number(((before + after) / 2).toFixed(6));
      repairs.push({
        frame, column: header[column], original: rows[i][column], estimated: value,
        method: "linear interpolation", units: "cm",
        neighbors: [{ frame: frame - 1, value: before }, { frame: frame + 1, value: after }],
      });
      rows[i][column] = String(value);
    }
  }
  return { csv: rows.map((row) => row.join(";")).join("\n") + "\n", frameCount: rows.length - 1, repairs };
}

export async function prepareDemo() {
  const directory = new URL("clean/", root);
  await mkdir(directory, { recursive: true });
  const report = { version: 1, encoding: "UTF-8", originalsPreserved: true, files: [] };
  for (const name of files) {
    const bytes = await readFile(new URL(name, root));
    const originalHash = createHash("sha256").update(bytes).digest("hex");
    const result = cleanCsv(new TextDecoder("windows-1251", { fatal: true }).decode(bytes), name);
    const cleanBytes = Buffer.from(result.csv, "utf8");
    const sha256 = createHash("sha256").update(cleanBytes).digest("hex");
    await writeFile(new URL(name, directory), cleanBytes);
    report.files.push({ name, originalSha256: originalHash, cleanSha256: sha256, frameCount: result.frameCount, repairs: result.repairs });
  }
  await writeFile(new URL("report.json", directory), JSON.stringify(report, null, 2) + "\n");
  console.log(`Validated ${report.files.length} CSV files; ${report.files.reduce((sum, file) => sum + file.repairs.length, 0)} estimated coordinates with provenance.`);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) await prepareDemo();
