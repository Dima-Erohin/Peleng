"""Этап детекции: синхронный проход по видео всех камер с кэшированием результатов в JSON.

Кэш: <out>/detections_<детектор>_<хэш-настроек>.json — {"config", "n_frames", "cams", "frames": [[[det...] на камеру] на кадр]}.
Повторный запуск с теми же настройками (и не меньшим числом кадров) берёт кэш — трекинг перезапускается за секунды.
Кадры обрабатываются строго по порядку (детектор видит только кадры <= k).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .detect_base import Detection
from .ue_data import SyncVideoReader, UEDataset


def cache_path(out_dir: Path, detector, ds: UEDataset) -> Path:
    cfg = {"det": detector.config() if hasattr(detector, "config") else repr(detector),
           "videos": [str(c.video_path.name) if c.video_path else None for c in ds.cameras],
           "data": str(Path(ds.root).resolve())}
    h = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]
    name = getattr(detector, "name", type(detector).__name__).replace("+", "_").replace(":", "_").replace("/", "_")
    return Path(out_dir) / f"detections_{name}_{h}.json"


def load_cache(path: Path, n_frames: int):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if d.get("n_frames", 0) < n_frames:
        return None
    return [[[Detection.from_list(x) for x in cam] for cam in fr] for fr in d["frames"][:n_frames]]


def load_detections_file(path, ds: UEDataset, n_frames: int) -> list:
    """Готовые детекции из JSON: {"cams": [id...]?, "frames": [[[det...] на камеру] на кадр]}.

    Детекция — список [x1, y1, x2, y2, u, v, score, cls, source] (как в кэше) или словарь
    {"bbox": [x1,y1,x2,y2], "point_px": [u,v]?, "score", "cls"?}. Порядок камер — "cams" (id) или как в наборе."""
    d = json.loads(Path(path).read_text())
    ids = d.get("cams") or [c.cam_id for c in ds.cameras]
    pos = {cid: i for i, cid in enumerate(ids)}
    out = []
    fr = d["frames"]
    for k in range(n_frames):
        row = []
        src = fr[k] if k < len(fr) else [[] for _ in ids]
        for c in ds.cameras:
            lst = src[pos[c.cam_id]] if c.cam_id in pos and pos[c.cam_id] < len(src) else []
            dets = []
            for x in lst:
                if isinstance(x, dict):
                    b = x["bbox"]
                    pt = x.get("point_px") or ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                    dets.append(Detection(tuple(b), tuple(pt), float(x.get("score", 1.0)), x.get("cls", "object"),
                                          x.get("source", "file")))
                else:
                    dets.append(Detection.from_list(x))
            row.append(dets)
        out.append(row)
    return out


def _log(msg: str) -> None:
    print(msg, flush=True)


def run_detection(ds: UEDataset, detector, n_frames: int, out_dir: Path | None = None, log=_log,
                  use_cache: bool = True) -> tuple[list, dict]:
    """Возвращает (frames, stats): frames[k][i] — детекции камеры ds.cameras[i] на кадре k."""
    cpath = cache_path(out_dir, detector, ds) if out_dir is not None else None
    if use_cache and cpath is not None:
        cached = load_cache(cpath, n_frames)
        if cached is not None:
            log(f"детекции: кэш {cpath.name}")
            return cached, {"cached": True, "cache": str(cpath), "seconds": 0.0}
    t0 = time.time()
    frames: list = []
    vids = [c.video_path for c in ds.cameras]
    if any(v is None for v in vids):
        raise FileNotFoundError("не для всех камер найдено видео")
    tdet = 0.0
    for k, imgs in SyncVideoReader(vids, n_frames):
        row = []
        for cam, img in zip(ds.cameras, imgs):
            if img is None:
                row.append([])
                continue
            t1 = time.time()
            row.append(detector.detect(cam.cam_id, k, img, cam.model(k)))
            tdet += time.time() - t1
        frames.append(row)
        if k % 50 == 0:
            el = time.time() - t0
            log(f"  кадр {k}/{n_frames}  {el:.0f} c  ({el / (k + 1):.2f} c/кадр, детектор {tdet / (k + 1):.2f})")
    stats = {"cached": False, "seconds": time.time() - t0, "detector_seconds": tdet, "frames": len(frames)}
    if cpath is not None:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        tmp = cpath.with_suffix(".tmp")
        tmp.write_text(json.dumps({"config": detector.config() if hasattr(detector, "config") else None,
                                   "n_frames": len(frames), "cams": [c.cam_id for c in ds.cameras],
                                   "frames": [[[d.to_list() for d in cam] for cam in fr] for fr in frames]}))
        tmp.replace(cpath)
        stats["cache"] = str(cpath)
    return frames, stats


if __name__ == "__main__":
    import argparse

    from .basic_detectors import make_detector
    from .ue_data import load_dataset

    ap = argparse.ArgumentParser(description="только этап детекции (с кэшем)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--detector", default="yolo+bg")
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args()
    ds = load_dataset(a.data)
    det = make_detector(a.detector)
    n = min(ds.n_frames, a.max_frames or ds.n_frames)
    _, st = run_detection(ds, det, n, Path(a.out))
    print(st)
