"""Оценка траектории против эталона: RMSE / медиана / максимум ошибки (3D и по осям), по числу камер и режимам.

Запуск отдельно:
    python offline/evaluate.py --est trajectory.csv --truth object_transform_SM_AA_track.csv [--diag trajectory_diag.csv]
Оба CSV — в формате эталона (Frame;Loc_X;Loc_Y;Loc_Z, сантиметры Unreal). Ошибки выводятся в метрах.
Дополнительно — разложение ошибки на продольную (вдоль курса эталона) и поперечную составляющие:
это показывает, сколько ошибки даёт выбор точки объекта (центр рамки против переда кабины).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "offline"

from .ue_data import read_ue_csv  # noqa: E402


def _stats(e: np.ndarray) -> dict:
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return {"n": 0}
    return {"n": int(len(e)), "rmse": float(np.sqrt(np.mean(e ** 2))), "median": float(np.median(np.abs(e))),
            "mean": float(np.mean(np.abs(e))), "p90": float(np.percentile(np.abs(e), 90)), "max": float(np.max(np.abs(e)))}


def evaluate(est_ue: np.ndarray, truth_ue: np.ndarray, n_cams: np.ndarray | None = None, modes: np.ndarray | None = None) -> dict:
    """est_ue, truth_ue: (N,3) в см Unreal (строка = кадр). Возвращает словарь метрик в метрах."""
    n = min(len(est_ue), len(truth_ue))
    E = np.asarray(est_ue[:n], dtype=np.float64) / 100.0
    T = np.asarray(truth_ue[:n], dtype=np.float64) / 100.0
    d = E - T
    err3 = np.linalg.norm(d, axis=1)
    # курс эталона (центральная разность — это только для анализа ошибки, не для оценки)
    vel = np.gradient(T[:, :2], axis=0)
    sp = np.linalg.norm(vel, axis=1)
    fwd = np.zeros((n, 2))
    ok = sp > 1e-3
    fwd[ok] = vel[ok] / sp[ok, None]
    along = np.where(ok, np.sum(d[:, :2] * fwd, axis=1), np.nan)
    cross = np.where(ok, d[:, 0] * -fwd[:, 1] + d[:, 1] * fwd[:, 0], np.nan)
    res = {
        "frames": int(n),
        "valid_frames": int(np.isfinite(err3).sum()),
        "err3d": _stats(err3),
        "err_horiz": _stats(np.linalg.norm(d[:, :2], axis=1)),
        "x": _stats(d[:, 0]), "y": _stats(d[:, 1]), "z": _stats(d[:, 2]),
        "along_track": {**_stats(along), "bias": float(np.nanmean(along)) if np.isfinite(along).any() else None},
        "cross_track": {**_stats(cross), "bias": float(np.nanmean(cross)) if np.isfinite(cross).any() else None},
        "z_bias": float(np.nanmean(d[:, 2])) if np.isfinite(d[:, 2]).any() else None,
    }
    if n_cams is not None:
        nc = np.asarray(n_cams[:n])
        res["by_ncams"] = {}
        for k in sorted(set(int(v) for v in nc)):
            lab = f"{k}" if k < 3 else "3+"
            m = nc == k if k < 3 else nc >= 3
            res["by_ncams"][lab] = _stats(err3[m])
    if modes is not None:
        md = np.asarray(modes[:n])
        res["by_mode"] = {str(m): _stats(err3[md == m]) for m in sorted(set(md.tolist()))}
    res["per_frame_err3d"] = err3
    return res


def format_report(r: dict) -> str:
    def row(name, s):
        if not s or s.get("n", 0) == 0:
            return f"  {name:<14} нет данных"
        return (f"  {name:<14} n={s['n']:5d}  RMSE={s['rmse']:8.2f}  медиана={s['median']:7.2f}  "
                f"p90={s['p90']:7.2f}  макс={s['max']:8.2f} м")

    lines = [f"Кадров: {r['frames']}, с оценкой: {r['valid_frames']}", row("3D", r["err3d"]), row("горизонт.", r["err_horiz"]),
             row("X", r["x"]), row("Y", r["y"]), row("Z", r["z"]),
             row("вдоль курса", r["along_track"]) + f"  (смещение {r['along_track'].get('bias') or 0:+.2f} м)",
             row("поперёк", r["cross_track"]) + f"  (смещение {r['cross_track'].get('bias') or 0:+.2f} м)",
             f"  смещение по Z: {r['z_bias'] or 0:+.2f} м"]
    if "by_ncams" in r:
        lines.append("По числу камер в обновлении:")
        lines += [row(f"камер {k}", s) for k, s in r["by_ncams"].items()]
    if "by_mode" in r:
        lines.append("По режиму:")
        lines += [row(k, s) for k, s in r["by_mode"].items()]
    return "\n".join(lines)


def to_jsonable(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "per_frame_err3d"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="оценка траектории против эталона")
    ap.add_argument("--est", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--diag", default=None, help="служебный CSV конвейера (число камер, режим)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    te = read_ue_csv(a.est, ("Loc_X", "Loc_Y", "Loc_Z"), interpolate=False)
    tt = read_ue_csv(a.truth, ("Loc_X", "Loc_Y", "Loc_Z"))
    E = np.stack([te["Loc_X"], te["Loc_Y"], te["Loc_Z"]], 1)
    T = np.stack([tt["Loc_X"], tt["Loc_Y"], tt["Loc_Z"]], 1)
    nc = md = None
    if a.diag:
        import csv

        rows = list(csv.DictReader(open(a.diag, encoding="utf-8"), delimiter=";"))
        nc = np.array([int(r["n_cams"]) for r in rows])
        md = np.array([r["mode"] for r in rows])
    r = evaluate(E, T, nc, md)
    print(format_report(r))
    if a.json:
        Path(a.json).write_text(json.dumps(to_jsonable(r), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
