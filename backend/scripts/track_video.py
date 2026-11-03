"""Export an explicitly seeded single-object 2D track from a local video."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peleng.video_tracking import TemplateTracker


def track_video(
    source: Path, seed_frame: int, roi: tuple[int, int, int, int],
    camera_id: str, object_id: str, threshold: float = 0.65, label: str = "drone",
) -> dict:
    if seed_frame < 0:
        raise ValueError("Seed frame must be nonnegative")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {source}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or total <= seed_frame or width < 1 or height < 1:
            raise ValueError("Invalid video timing or seed is past the end")
        capture.set(cv2.CAP_PROP_POS_FRAMES, seed_frame)
        ok, seed = capture.read()
        if not ok:
            raise ValueError(f"Cannot decode seed frame {seed_frame}")
        forward = TemplateTracker(seed, roi, threshold=threshold)
        backward = TemplateTracker(seed, roi, threshold=threshold)
        records = [None] * total
        records[seed_frame] = {
            "frame": seed_frame, "time": seed_frame / fps, "bbox": list(roi),
            "status": "seed", "correlation": 1.0, "processing_ms": 0.0,
        }
        # Sequential forward decode; reverse traversal uses seeking, avoiding
        # storing hundreds of full-resolution decoded frames in RAM.
        for index in range(seed_frame + 1, total):
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Decode failure at frame {index} of {total}")
            match = forward.update(image)
            records[index] = record(index, fps, match)
        for index in range(seed_frame - 1, -1, -1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Decode failure at frame {index} of {total}")
            records[index] = record(index, fps, backward.update(image))
    finally:
        capture.release()
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    visible = sum(item["bbox"] is not None for item in records)
    return {
        "version": 1, "space": "image", "camera_id": camera_id, "object_id": object_id,
        "label": label, "method": "OpenCV fixed-seed multiscale normalized correlation",
        "source": source.name, "source_sha256": digest,
        "width": width, "height": height, "fps": fps, "frame_count": total,
        "seed_frame": seed_frame, "seed_bbox": list(roi), "threshold": threshold,
        "visible_frames": visible, "lost_frames": total - visible,
        "note": "User-identified object. 2D similarity only; not semantic detection or world XYZ.",
        "frames": records,
    }


def record(index, fps, match) -> dict:
    return {
        "frame": index, "time": index / fps,
        "bbox": list(match.bbox) if match.bbox else None,
        "status": "tracked" if match.bbox else "lost",
        "correlation": round(match.correlation, 4),
        "peak_margin": round(match.peak_margin, 4),
        "processing_ms": round(match.processing_ms, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--seed-frame", type=int, required=True)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), required=True)
    parser.add_argument("--camera-id", default="cam-5")
    parser.add_argument("--object-id", default="drone-1")
    parser.add_argument("--label", default="drone")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.video.resolve():
        parser.error("Output must not overwrite the source video")
    try:
        result = track_video(args.video, args.seed_frame, tuple(args.roi), args.camera_id, args.object_id, args.threshold, args.label)
    except (ValueError, OSError, cv2.error) as error:
        parser.exit(1, f"Tracking failed: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.output.parent,
        prefix=f".{args.output.name}.", suffix=".tmp", delete=False,
    )
    temporary = Path(stream.name)
    try:
        with stream:
            json.dump(result, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{result['visible_frames']}/{result['frame_count']} frames matched; {result['lost_frames']} explicitly lost")
    print(args.output)


if __name__ == "__main__":
    main()
