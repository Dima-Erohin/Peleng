"""Точка входа офлайн-конвейера: один запуск -> траектория CSV + служебный CSV + графики + метрики.

    python offline/run.py --data <папка с mp4 и csv> --out <папка результатов> [--detector yolo+bg] [--max-frames N]

Шаги: загрузка набора -> детекция во всех камерах (кэш в <out>) -> причинный 3D-трекинг -> запись
trajectory.csv (формат эталона: Frame;Loc_X;Loc_Y;Loc_Z, см Unreal) -> оценка и графики, если в папке есть эталон.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "offline"

from .detect_stage import run_detection  # noqa: E402
from .tracker3d import Tracker3D, TrackerConfig  # noqa: E402
from .ue_data import load_dataset, world_to_ue, write_ue_csv  # noqa: E402


def track(ds, frames: list, cfg: TrackerConfig, n: int):
    """Причинный трекинг: кадр за кадром, выход кадра k зависит только от детекций кадров <= k."""
    tr = Tracker3D(cfg)
    ids = [c.cam_id for c in ds.cameras]
    out = []
    for k in range(n):
        cams = [c.model(k) for c in ds.cameras]
        out.append(tr.step(k, cams, frames[k], ids))
    return out, tr


def results_to_arrays(res):
    n = len(res)
    P = np.full((n, 3), np.nan)
    for i, r in enumerate(res):
        if r.pos is not None:
            P[i] = r.pos
    return world_to_ue(P), np.array([r.n_cams for r in res]), np.array([r.mode for r in res]), np.array([r.sigma_m for r in res])


def write_diag(path: Path, res, P_ue) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("Frame;mode;n_cams;cams;sigma_m;reproj_px;speed_mps;Loc_X;Loc_Y;Loc_Z\r\n")
        for r, p in zip(res, P_ue):
            sp = float(np.linalg.norm(r.vel)) if r.vel is not None else float("nan")
            cams = ",".join(str(c) for c in r.cams)
            xyz = ";".join("" if not np.isfinite(v) else f"{v:.4f}" for v in p)
            f.write(f"{r.frame};{r.mode};{r.n_cams};{cams};{r.sigma_m:.3f};{r.reproj_px:.2f};{sp:.2f};{xyz}\r\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_rev() -> str | None:
    try:
        root = Path(__file__).resolve().parents[1]
        rev = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "offline"], capture_output=True, text=True,
                               timeout=5).stdout.strip()
        return rev + ("+изменения" if dirty else "") if rev else None
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="3D-траектория объекта по видео нескольких камер (причинная обработка)")
    ap.add_argument("--data", required=True, help="папка с видео *.mp4 и CSV камер object_transform_*N.csv")
    ap.add_argument("--out", required=True, help="папка результатов")
    ap.add_argument("--detector", default="yolo+bg", help="yolo+bg | yolo | bg | модуль:Класс")
    ap.add_argument("--yolo-weights", default="yolo11n.pt")
    ap.add_argument("--yolo-classes", default="truck,car,bus", help="через запятую; пусто — все классы")
    ap.add_argument("--yolo-imgsz", type=int, default=640)
    ap.add_argument("--yolo-conf", type=float, default=0.15)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true", help="не брать детекции из кэша")
    ap.add_argument("--point", default="center", choices=["center", "bottom", "top", "detector"],
                    help="какую точку рамки триангулировать")
    ap.add_argument("--model", default="cv", choices=["cv", "ca"], help="модель движения: постоянная скорость / ускорение")
    ap.add_argument("--accel-std", type=float, default=3.0, help="σ ускорения (cv), м/с²")
    ap.add_argument("--no-size-range", action="store_true", help="не использовать дальность по размеру рамки при одной камере")
    ap.add_argument("--offset-fwd", type=float, default=0.0, help="сдвиг выходной точки вперёд по курсу, м")
    ap.add_argument("--offset-up", type=float, default=0.0, help="сдвиг выходной точки вверх, м")
    ap.add_argument("--truth", default=None, help="имя CSV эталона в папке данных (по умолчанию — найти автоматически)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--tag", default="", help="подпись для графиков")
    return ap


def make_detector_from_args(a):
    from .basic_detectors import make_detector

    classes = tuple(c.strip() for c in a.yolo_classes.split(",") if c.strip())
    ykw = dict(weights=a.yolo_weights, classes=classes, imgsz=a.yolo_imgsz, conf=a.yolo_conf)
    if a.detector == "yolo":
        return make_detector("yolo", **ykw)
    if a.detector in ("yolo+bg", "bg+yolo"):
        return make_detector("yolo+bg", **{"yolo_" + k: v for k, v in ykw.items()})
    return make_detector(a.detector)


def main(argv=None) -> dict:
    a = build_parser().parse_args(argv)
    t_start = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(a.data, a.truth)
    n = ds.n_frames if a.max_frames is None else min(ds.n_frames, a.max_frames)
    print(f"набор: {len(ds.cameras)} камер {[c.cam_id for c in ds.cameras]}, кадров {n}, fps {ds.fps:.2f}, "
          f"эталон: {'есть' if ds.truth_ue is not None else 'нет'}", flush=True)

    det = make_detector_from_args(a)
    frames, dstats = run_detection(ds, det, n, out, use_cache=not a.no_cache)
    t_det = time.time()

    cfg = TrackerConfig(fps=ds.fps, model=a.model, accel_std=a.accel_std, point=a.point,
                        size_range=not a.no_size_range, out_offset_fwd_m=a.offset_fwd, out_offset_up_m=a.offset_up)
    res, _ = track(ds, frames, cfg, n)
    t_trk = time.time()
    P_ue, n_cams, modes, sig = results_to_arrays(res)

    traj = out / "trajectory.csv"
    write_ue_csv(traj, np.arange(n), P_ue)
    write_diag(out / "trajectory_diag.csv", res, P_ue)
    uniq, cnt = np.unique(modes, return_counts=True)
    info = {
        "script": "offline/run.py", "args": vars(a), "git": _git_rev(), "python": platform.python_version(),
        "frames": n, "cameras": [c.cam_id for c in ds.cameras], "detector": det.config() if hasattr(det, "config") else str(det),
        "tracker": cfg.__dict__, "detection": dstats, "modes": dict(zip(uniq.tolist(), cnt.tolist())),
        "time_s": {"detection": round(t_det - t_start, 1), "tracking": round(t_trk - t_det, 2)},
    }
    print(f"режимы: {info['modes']}; детекция {info['time_s']['detection']} c, трекинг {info['time_s']['tracking']} c", flush=True)

    metrics = None
    if ds.truth_ue is not None:
        from .evaluate import evaluate, format_report, to_jsonable

        metrics = evaluate(P_ue, ds.truth_ue[:n], n_cams, modes)
        rep = format_report(metrics)
        print(rep)
        (out / "metrics.txt").write_text(rep + "\n", encoding="utf-8")
        (out / "metrics.json").write_text(json.dumps(to_jsonable(metrics), ensure_ascii=False, indent=1), encoding="utf-8")
        info["metrics_summary"] = {"rmse3d_m": metrics["err3d"].get("rmse"), "median3d_m": metrics["err3d"].get("median")}
    if not a.no_plots:
        from .plots import make_plots

        paths = make_plots(out, P_ue, None if ds.truth_ue is None else ds.truth_ue[:n], modes, n_cams, sig, a.tag)
        info["plots"] = [str(p) for p in paths]
    info["time_s"]["total"] = round(time.time() - t_start, 1)
    info["trajectory_sha256"] = _sha256(traj)
    (out / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"готово: {traj}  ({info['time_s']['total']} c)", flush=True)
    return {"info": info, "metrics": metrics, "results": res}


if __name__ == "__main__":
    main()
