export function parseImageTrack(value) {
  if (value?.version !== 1 || value.space !== "image" || !value.camera_id || !value.object_id ||
      !Number.isInteger(value.width) || value.width < 1 || !Number.isInteger(value.height) || value.height < 1 ||
      !Number.isFinite(value.fps) || value.fps <= 0 ||
      !Number.isInteger(value.frame_count) || value.frame_count < 1 || value.frame_count > 200_000 ||
      !Array.isArray(value.frames) || value.frames.length !== value.frame_count) {
    throw new Error("Некорректный формат 2D-трека");
  }
  value.frames.forEach((frame, i) => {
    if (frame.frame !== i || !Number.isFinite(frame.time) || Math.abs(frame.time - i / value.fps) > 0.0001 ||
        !["seed", "tracked", "lost"].includes(frame.status) ||
        !Number.isFinite(frame.correlation) || frame.correlation < -1 || frame.correlation > 1) {
      throw new Error(`2D-трек: неверный отсчёт ${i}`);
    }
    if (frame.status === "lost") {
      if (frame.bbox !== null) throw new Error(`2D-трек: у пропуска ${i} не должно быть рамки`);
    } else {
      const box = frame.bbox;
      if (!Array.isArray(box) || box.length !== 4 || box.some((n) => !Number.isFinite(n)) ||
          box[0] < 0 || box[1] < 0 || box[2] <= 0 || box[3] <= 0 ||
          box[0] + box[2] > value.width || box[1] + box[3] > value.height) {
        throw new Error(`2D-трек: неверная рамка ${i}`);
      }
    }
  });
  return value;
}

export function sampleImageTrack(track, time) {
  if (!Number.isFinite(time) || time < 0 || time > track.frame_count / track.fps) return null;
  return track.frames[Math.min(track.frames.length - 1, Math.floor(time * track.fps + 0.00001))];
}
