"""Точка входа офлайн-конвейера: один запуск -> CSV траекторий объектов + служебные файлы + графики + метрики.

    python offline/run.py --data <папка с mp4 и csv> --out <папка результатов> [--objects truck,drone] [--max-frames N]

Для каждого объекта (пресеты — offline/presets.py): детекция во всех камерах (кэш в <out>/<объект>/) ->
причинный 3D-трекинг -> <out>/<объект>.csv (формат эталона: Frame;Loc_X;Loc_Y;Loc_Z, см Unreal) ->
служебный CSV, графики, метрики (если есть эталон и объект с ним сравнивается) -> <out>/viewer.json для фронтенда.
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
from .tracker3d import FrameResult, Tracker3D, TrackerConfig  # noqa: E402
from .ue_data import load_dataset, world_to_ue, write_ue_csv  # noqa: E402


class _Single:
    """Обёртка одиночного трекера под общий интерфейс с MultiTracker3D."""

    def __init__(self, cfg):
        self.tr = Tracker3D(cfg)

    def trackers(self):
        return [self.tr]

    def step(self, k, cams, dets, ids):
        return {1: self.tr.step(k, cams, dets, ids)}

    def primary(self, res):
        return 1


def track(ds, frames: list, cfg: TrackerConfig, n: int, refiner=None, log=print, multi: int = 0,
          refiner_only_if_empty: bool = False, refiner_full_every: int = 0, refiner_full_imgsz: int = 1280):
    """Причинный трекинг: кадр за кадром, выход кадра k зависит только от детекций кадров <= k.

    refiner (RoiRefiner) — сильный детектор в окне вокруг прогноза трекера (прогноз — по кадрам < k).
    multi > 0 — несколько объектов (MultiTracker3D, не больше multi треков); основной объект — в первом результате.
    Возвращает (results основного объекта по кадрам, трекер, детекции по кадрам, все треки {кадр: {id: FrameResult}})."""
    from .tracker3d import MultiTracker3D

    mt = MultiTracker3D(cfg, max_tracks=multi) if multi > 0 else _Single(cfg)
    ids = [c.cam_id for c in ds.cameras]
    out, used_dets, all_tracks = [], [], []
    reader = None
    if refiner is not None:
        from .roi_refine import merge
        from .ue_data import SyncVideoReader

        reader = iter(SyncVideoReader([c.video_path for c in ds.cameras], n))
    t0 = time.time()
    for k in range(n):
        cams = [c.model(k) for c in ds.cameras]
        row = frames[k]
        if reader is not None:
            _, imgs = next(reader)
            trs = mt.trackers() if hasattr(mt, "trackers") else [t for _, t in mt.tracks]
            regs = [t.predict_regions(cams) for t in trs]
            lost = not trs or all(t._lost() for t in trs)
            row = []
            for i, (cam, img) in enumerate(zip(cams, imgs)):
                base = frames[k][i]
                wins = [refiner.window(r[i], cam.size) for r in regs if r[i] is not None]
                if refiner_only_if_empty and base:
                    wins = []  # основной детектор уже нашёл объект в этой камере — окно не нужно
                elif lost or multi > 0:
                    # трек потерян (или ищем новые объекты): окна вокруг лучших кандидатов базового детектора
                    for d in sorted(base, key=lambda d: -d.score)[:1]:
                        if d.score >= 0.25:
                            wins.append(refiner.window(d.bbox, cam.size))
                roi = []
                if img is not None:
                    for w in dict.fromkeys(wins):
                        roi += refiner.detect(ids[i], k, img, w)
                    # трека нет/потерян и основной детектор пуст (например, камера движется и вычитание фона
                    # отключено): раз в full_every кадров — сильная модель на весь кадр
                    if lost and not base and refiner_full_every and k % refiner_full_every == 0:
                        W, H = cam.size
                        roi += refiner.detect(ids[i], k, img, (0, 0, max(W, H)), imgsz=refiner_full_imgsz)
                row.append(merge(base, roi))
            if k % 100 == 0:
                log(f"  трекинг+ROI: кадр {k}/{n}, {time.time() - t0:.0f} c, окон: сеть {refiner.n_run}, кэш {refiner.n_cached}")
        res = mt.step(k, cams, row, ids)
        pid = mt.primary(res)
        if pid is None:
            out.append(FrameResult(k, None, None, "none", 0, [], float("inf")))
        else:
            r = res[pid]
            r.info["track_id"] = pid
            out.append(r)
        all_tracks.append(res)
        used_dets.append(row)
    if refiner is not None:
        refiner.save()
    return out, mt, used_dets, all_tracks


def results_to_arrays(res):
    n = len(res)
    P = np.full((n, 3), np.nan)
    for i, r in enumerate(res):
        if r.pos is not None:
            P[i] = r.pos
    return world_to_ue(P), np.array([r.n_cams for r in res]), np.array([r.mode for r in res]), np.array([r.sigma_m for r in res])


def write_diag(path: Path, res, P_ue) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("Frame;mode;n_cams;cams;used_det;sigma_m;reproj_px;speed_mps;Loc_X;Loc_Y;Loc_Z;limited_update\r\n")
        for r, p in zip(res, P_ue):
            sp = float(np.linalg.norm(r.vel)) if r.vel is not None else float("nan")
            cams = ",".join(str(c) for c in r.cams)
            # used_det: «индекс камеры в наборе:индекс детекции» — какие детекции вошли в обновление
            used = ",".join(f"{i}:{j}" for i, j in r.info.get("used_det", []))
            xyz = ";".join("" if not np.isfinite(v) else f"{v:.4f}" for v in p)
            f.write(f"{r.frame};{r.mode};{r.n_cams};{cams};{used};{r.sigma_m:.3f};{r.reproj_px:.2f};{sp:.2f};{xyz};{int(bool(r.info.get('limited_update')))}\r\n")

def video_observations(res, detections, cameras, fps):
    return {
        "fps": fps, "sizes": [list(camera.model(0).size) for camera in cameras],
        "cams": [camera.cam_id for camera in cameras],
        "points": [{
            "frame": r.frame, "t": r.frame / fps, "mode": r.mode, "n_cams": r.n_cams,
            "sigma_m": float(r.sigma_m) if np.isfinite(r.sigma_m) else None,
            "limited_update": bool(r.info.get("limited_update")), "used_det": r.info.get("used_det", []),
            "dets": [[[round(float(v), 1) for v in d.bbox] + [round(float(d.score), 3), d.source]
                     for d in row] for row in detections[r.frame]],
        } for r in res],
    }


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
    ap = argparse.ArgumentParser(description="3D-траектории объектов по видео нескольких камер (причинная обработка)")
    ap.add_argument("--data", required=True, help="папка с видео *.mp4 и CSV камер object_transform_*N.csv")
    ap.add_argument("--out", required=True, help="папка результатов")
    ap.add_argument("--objects", default="truck,drone", help="пресеты объектов через запятую (offline/presets.py)")
    ap.add_argument("--objects-file", default=None, help="JSON со списком описаний объектов (вместо --objects)")
    ap.add_argument("--set", action="append", default=[], metavar="ОБЪЕКТ.ключ=значение",
                    help="переопределить поле описания: truck.tracker.accel_std=2, drone.detections=/путь.json")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true", help="не брать детекции из кэша")
    ap.add_argument("--truth", default=None, help="имя CSV эталона в папке данных (по умолчанию — найти автоматически)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-viewer", action="store_true", help="не писать viewer.json")
    ap.add_argument("--no-road", action="store_true", help="не добавлять в viewer.json дорогу (road)")
    ap.add_argument("--strict", action="store_true",
                    help="падать при ошибке обработки объекта (по умолчанию — объект пропускается с сообщением, "
                         "остальные объекты обрабатываются)")
    ap.add_argument("--skip-missing", action="store_true", help="(устарело; теперь поведение по умолчанию)")
    return ap


def fuse_frames(parts: list, n: int, primary_sources, mode: str = "fallback", drop_border_px: float = 3.0,
                point: str = "kp:pivot") -> list:
    """Слияние детекций нескольких детекторов по кадрам и камерам.

    primary — детекции из primary_sources (например, поза грузовика с точкой pivot); рамка у края кадра у них
    отбрасывается (ключевая точка может оказаться за кадром). mode="fallback": в камере берутся основные, если они
    есть, иначе запасные; mode="all": все вместе."""
    out = []
    for k in range(n):
        row = []
        for i in range(len(parts[0][k])):
            prim, sec = [], []
            for fr in parts:
                for d in fr[k][i]:
                    (prim if d.source in primary_sources else sec).append(d)
            prim = [d for d in prim if d.point_is_reliable(point)]
            if drop_border_px is not None and prim:
                prim = [d for d in prim if d.bbox[0] > drop_border_px and d.bbox[1] > drop_border_px
                        and d.bbox[2] < d.extra.get("img_w", 1920) - drop_border_px
                        and d.bbox[3] < d.extra.get("img_h", 1080) - drop_border_px]
            row.append(prim if (mode == "fallback" and prim) else prim + sec)
        out.append(row)
    return out


def build_detector_frames(ds, spec: dict, n: int, out: Path, use_cache: bool):
    """Детекции объекта по кадрам: из файла, из кэша или запуском детектора (или нескольких — "detectors" +
    слияние). Возвращает (frames, stats, config)."""
    if spec.get("detectors"):
        parts, stats, cfgs = [], [], []
        for sub in spec["detectors"]:
            sspec = {**spec, **sub}
            sspec.pop("detectors", None)
            try:
                f, st, cf = build_detector_frames(ds, sspec, n, out, use_cache)
            except Exception as e:  # noqa: BLE001 — нет весов и т. п.: работаем без этого детектора
                if sub.get("optional"):
                    print(f"детектор {sub.get('detector')} недоступен ({type(e).__name__}: {e}) — без него", flush=True)
                    continue
                raise
            parts.append(f)
            stats.append(st)
            cfgs.append(cf)
        if not parts:
            raise RuntimeError("ни один детектор не доступен")
        prim = tuple((spec.get("tracker") or {}).get("primary_sources", ()))
        fu = spec.get("fusion") or {}
        frames = fuse_frames(parts, n, prim, fu.get("mode", "fallback"), fu.get("drop_border_px", 3.0),
                             (spec.get("tracker") or {}).get("point", "center"))
        return frames, {"parts": stats}, {"parts": cfgs, "fusion": fu}
    from .basic_detectors import combine_dets, make_detector
    from .detect_stage import load_detections_file

    if spec.get("detections"):
        return load_detections_file(spec["detections"], ds, n), {"file": str(spec["detections"])}, {"file": str(spec["detections"])}
    name = spec["detector"]
    kw = dict(spec.get("detector_kw") or {})
    if name in ("yolo+bg", "bg+yolo"):
        ykw = {k[5:]: v for k, v in kw.items() if k.startswith("yolo_")}
        bkw = {k[3:]: v for k, v in kw.items() if k.startswith("bg_")}
        if "classes" in ykw:
            ykw["classes"] = tuple(ykw["classes"])
        dy, db = make_detector("yolo", **ykw), make_detector("bg", **bkw)
        # два прохода с отдельными кэшами: можно менять один детектор, не пересчитывая другой
        fy, sy = run_detection(ds, dy, n, out, use_cache=use_cache)
        fb, sb = run_detection(ds, db, n, out, use_cache=use_cache)
        frames = [[combine_dets(y, b) for y, b in zip(ry, rb)] for ry, rb in zip(fy, fb)]
        return frames, {"yolo": sy, "bg": sb, "seconds": sy.get("seconds", 0) + sb.get("seconds", 0)}, \
            {"yolo": dy.config(), "bg": db.config()}
    if name in ("yolo", "bg") and "classes" in kw:
        kw["classes"] = tuple(kw["classes"])
    det = make_detector(name, **kw)
    frames, st = run_detection(ds, det, n, out, use_cache=use_cache)
    return frames, st, det.config() if hasattr(det, "config") else {"name": name}


def run_object(ds, spec: dict, n: int, out_root: Path, a) -> dict:
    """Полный цикл для одного объекта. Возвращает сводку (пути, режимы, метрики)."""
    name = spec["name"]
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"=== объект {name}: детектор {spec.get('detections') or spec['detector']}", flush=True)
    frames, dstats, dcfg = build_detector_frames(ds, spec, n, out, not a.no_cache)
    t_det = time.time()
    tk = dict(spec.get("tracker") or {})
    tk.setdefault("fps", ds.fps)
    tk["fps"] = ds.fps
    cfg = TrackerConfig(**tk)
    refiner = None
    if spec.get("roi"):
        from .roi_refine import RoiRefiner

        r = spec["roi"]
        refiner = RoiRefiner(r.get("weights", "yolo11m.pt"), imgsz=r.get("imgsz", 640), classes=tuple(r.get("classes", ())),
                             conf=r.get("conf", 0.1),
                             cache_path=out / f"roi_cache_{Path(r.get('weights', 'yolo11m.pt')).stem}_{r.get('imgsz', 640)}.json",
                             device=r.get("device", "cpu"))
    res, _, used, all_tracks = track(ds, frames, cfg, n, refiner, log=lambda m: print(m, flush=True),
                                     multi=int(spec.get("multi", 0)),
                                     refiner_only_if_empty=bool((spec.get("roi") or {}).get("only_if_empty", False)),
                                     refiner_full_every=int((spec.get("roi") or {}).get("full_every", 0)),
                                     refiner_full_imgsz=int((spec.get("roi") or {}).get("full_imgsz", 1280)))
    t_trk = time.time()
    (out / "detections_used.json").write_text(json.dumps({
        "note": "детекции, поданные в трекер, frames[k][i] -> камера cams[i]",
        "cams": [c.cam_id for c in ds.cameras], "n_frames": n,
        "frames": [[[d.to_list() for d in cam] for cam in fr] for fr in used]}))
    P_ue, n_cams, modes, sig = results_to_arrays(res)
    if spec.get("fill_before_init", "first") == "first":
        ok = np.where(np.isfinite(P_ue[:, 0]))[0]
        if len(ok) and ok[0] > 0:
            # единственное отступление ради полноты файла: до первой оценки объект не наблюдался —
            # пишем первую оценку (в служебном файле эти кадры помечены mode=none)
            P_ue[: ok[0]] = P_ue[ok[0]]
    csv_path = out_root / spec.get("out_csv", f"{name}.csv")
    write_ue_csv(csv_path, np.arange(n), P_ue)
    write_diag(out_root / (Path(spec.get("out_csv", f"{name}.csv")).stem + "_diag.csv"), res, P_ue)
    observations_path = out_root / (Path(spec.get("out_csv", f"{name}.csv")).stem + "_observations.json")
    observations = video_observations(res, used, ds.cameras, ds.fps)
    observations["trajectory_sha256"] = hashlib.sha256(csv_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    observations_path.write_text(json.dumps(observations,
                                            ensure_ascii=False, allow_nan=False), encoding="utf-8")
    uniq, cnt = np.unique(modes, return_counts=True)
    summary = {
        "name": name, "csv": str(csv_path), "detector": dcfg, "tracker": cfg.__dict__, "detection": dstats,
        "modes": dict(zip(uniq.tolist(), cnt.tolist())),
        "time_s": {"detection": round(t_det - t0, 1), "tracking": round(t_trk - t_det, 2)},
        "reproj_px_median": float(np.nanmedian([r.reproj_px for r in res if r.mode == "tri"])) if any(r.mode == "tri" for r in res) else None,
    }
    # согласованность без эталона: плавность (ускорение по 2-й разности) и ошибка репроекции
    Pw = P_ue / 100.0
    acc = np.linalg.norm(np.diff(Pw, 2, axis=0), axis=1) * ds.fps ** 2
    summary["smoothness"] = {"accel_median_mps2": float(np.nanmedian(acc)) if np.isfinite(acc).any() else None,
                             "accel_p95_mps2": float(np.nanpercentile(acc, 95)) if np.isfinite(acc).any() else None}
    if refiner is not None:
        summary["roi"] = {**refiner.config(), "runs": refiner.n_run, "cached": refiner.n_cached, "model_s": round(refiner.t_model, 1)}
    print(f"режимы: {summary['modes']}; детекция {summary['time_s']['detection']} c, трекинг {summary['time_s']['tracking']} c",
          flush=True)
    truth = ds.truth_ue[:n] if (spec.get("truth") and ds.truth_ue is not None) else None
    if truth is not None:
        from .evaluate import evaluate, format_report, to_jsonable

        metrics = evaluate(P_ue, truth, n_cams, modes)
        rep = format_report(metrics)
        print(rep, flush=True)
        (out / "metrics.txt").write_text(rep + "\n", encoding="utf-8")
        (out / "metrics.json").write_text(json.dumps(to_jsonable(metrics), ensure_ascii=False, indent=1), encoding="utf-8")
        summary["metrics"] = {"rmse3d_m": metrics["err3d"].get("rmse"), "median3d_m": metrics["err3d"].get("median"),
                              "by_ncams": metrics.get("by_ncams")}
    if not a.no_plots:
        from .plots import make_plots

        summary["plots"] = [str(p) for p in make_plots(out, P_ue, truth, modes, n_cams, sig, name)]
    summary["sha256"] = _sha256(csv_path)
    summary["_P_ue"] = P_ue
    summary["_res"] = res
    summary["_road"] = spec.get("road")
    return summary


def main(argv=None) -> dict:
    from .presets import get_specs

    a = build_parser().parse_args(argv)
    t_start = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(a.data, a.truth)
    n = ds.n_frames if a.max_frames is None else min(ds.n_frames, a.max_frames)
    print(f"набор: {len(ds.cameras)} камер {[c.cam_id for c in ds.cameras]}, кадров {n}, fps {ds.fps:.2f}, "
          f"эталон: {'есть' if ds.truth_ue is not None else 'нет'}", flush=True)
    specs = get_specs([x.strip() for x in a.objects.split(",") if x.strip()], a.objects_file, a.set)
    results = []
    for spec in specs:
        try:
            results.append(run_object(ds, spec, n, out, a))
        except Exception as e:  # noqa: BLE001 — один объект не должен ронять весь прогон
            if a.strict:
                raise
            import traceback

            traceback.print_exc()
            print(f"!!! объект {spec['name']} пропущен: {type(e).__name__}: {e}", flush=True)
    info = {
        "script": "offline/run.py", "argv": sys.argv[1:] if argv is None else argv, "git": _git_rev(),
        "python": platform.python_version(), "frames": n, "cameras": [c.cam_id for c in ds.cameras],
        "objects": [{k: v for k, v in r.items() if not k.startswith("_")} for r in results],
        "time_s": round(time.time() - t_start, 1),
    }
    (out / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if not a.no_viewer and results:
        from .export_viewer import build_multi

        vj = build_multi(ds, out, results, n, road=not a.no_road)
        (out / "viewer.json").write_text(json.dumps(vj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    for r in results:
        print(f"готово: {r['csv']}" + (f"  RMSE {r['metrics']['rmse3d_m']:.2f} м" if r.get("metrics") else ""), flush=True)
    print(f"всего {info['time_s']} c", flush=True)
    return {"info": info, "results": results}


if __name__ == "__main__":
    main()
