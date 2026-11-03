"""Дорога для фронтенда: осевая линия = сглаженная траектория грузовика (без ИИ: грузовик едет по центру дороги).

    python offline/road.py --viewer <out>/viewer.json [--width-m 4] [--out viewer.json] [--object truck]

Что делает build_road():
  1. берёт оценку траектории (наша, а не эталон), кадры в режиме `none` отбрасывает; `predict` длиннее
     max_predict_s считает «uncertain» (линия там остаётся, но помечена);
  2. выбросы — фильтр Хампеля по скользящей медиане (по каждой координате);
  3. прореживание по длине дуги (шаг step_m, ~2 м) + гауссово сглаживание по длине дуги (smooth_m) —
     НЕкаузально, это визуализация, а не сдаваемый CSV;
  4. высота поверхности = высота pivot − h_pivot_m (pivot модели грузовика выше земли; константа
     TRUCK_H_PIVOT_M в presets.py и там же описано, как получена);
  5. кромки left/right = центр ± width_m/2 по нормали к курсу в горизонтальной плоскости.

Система координат — та же, что у `three_m` в viewer.json (метры, правая, Y вверх: (x,y,z) = (X_ue, Z_ue, Y_ue)/100),
чтобы дорога ложилась на траекторию без пересчётов. Копия в Unreal (см) лежит в road.ue_cm.
«Левая» кромка — слева по ходу движения (в Three.js: left = (fz, 0, −fx), f — курс).
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

from .presets import TRUCK_H_PIVOT_M, TRUCK_WIDTH_M  # noqa: E402

TRUSTED_MODES = ("tri", "mono", "mono_size_prior", "init")
DEFAULT_ROAD = {"enabled": True, "h_pivot_m": TRUCK_H_PIVOT_M, "truck_width_m": TRUCK_WIDTH_M, "width_factor": 1.6,
                "width_m": None, "step_m": 2.0, "smooth_m": 10.0, "max_predict_s": 1.0}


def _ue_to_three(P_ue: np.ndarray) -> np.ndarray:
    P = np.asarray(P_ue, dtype=np.float64)
    return np.stack([P[:, 0], P[:, 2], P[:, 1]], 1) / 100.0


def _three_to_ue(P: np.ndarray) -> np.ndarray:
    return np.stack([P[:, 0], P[:, 2], P[:, 1]], 1) * 100.0


def _hampel(P: np.ndarray, win: int, k: float = 3.0, floor_m: float = 1.0) -> np.ndarray:
    """Маска «не выброс» по каждой из координат относительно скользящей медианы (окно win кадров)."""
    n, h = len(P), win // 2
    ok = np.ones(n, bool)
    for i in range(n):
        seg = P[max(0, i - h): i + h + 1]
        med = np.median(seg, 0)
        mad = 1.4826 * np.median(np.abs(seg - med), 0)
        if np.any(np.abs(P[i] - med) > np.maximum(k * mad, floor_m)):
            ok[i] = False
    return ok


def _gauss_smooth(P: np.ndarray, step_m: float, sigma_m: float) -> np.ndarray:
    """Гауссово сглаживание равномерно прореженной линии (нормировка ядра на краях — без «завала» к нулю)."""
    if sigma_m <= 0 or len(P) < 3:
        return P.copy()
    sig = sigma_m / step_m
    r = int(np.ceil(3 * sig))
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
    out = np.empty_like(P)
    for i in range(len(P)):
        a, b = max(0, i - r), min(len(P), i + r + 1)
        ww = w[a - i + r: b - i + r]
        out[i] = (P[a:b] * ww[:, None]).sum(0) / ww.sum()
    return out


def _segments(mask: np.ndarray) -> list[list[int]]:
    seg, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            seg.append([i, j])
            i = j + 1
        else:
            i += 1
    return seg


def centerline_from_track(P_three: np.ndarray, modes=None, fps: float = 30.0, *, h_pivot_m: float = TRUCK_H_PIVOT_M,
                          step_m: float = 2.0, smooth_m: float = 10.0, max_predict_s: float = 1.0):
    """(N,3) оценка pivot в «three» (м) + режимы по кадрам -> (centerline (M,3), uncertain bool (M,)) или None."""
    P = np.asarray(P_three, dtype=np.float64)
    n = len(P)
    valid = np.all(np.isfinite(P), 1)
    unc = np.zeros(n, bool)
    if modes is not None:
        modes = list(modes)
        valid &= np.array([m != "none" for m in modes])
        i = 0
        while i < n:  # длинные серии predict -> uncertain
            if modes[i] == "predict":
                j = i
                while j + 1 < n and modes[j + 1] == "predict":
                    j += 1
                if (j - i + 1) > max_predict_s * fps:
                    unc[i: j + 1] = True
                i = j + 1
            else:
                i += 1
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return None
    ok = _hampel(P[idx], win=max(5, int(fps)) | 1)
    idx = idx[ok]
    if len(idx) < 2:
        return None
    Q, U = P[idx], unc[idx]
    d = np.hypot(np.diff(Q[:, 0]), np.diff(Q[:, 2]))
    s = np.concatenate([[0.0], np.cumsum(d)])
    # стоянка: одинаковые s — оставляем первую точку, флаги объединяем
    keep = np.concatenate([[True], d > 1e-6])
    grp = np.cumsum(keep) - 1
    Uk = np.zeros(keep.sum(), bool)
    np.logical_or.at(Uk, grp, U)
    s, Q = s[keep], Q[keep]
    if len(s) < 2:
        return None
    grid = np.arange(0.0, s[-1] + 1e-9, step_m)
    if grid[-1] < s[-1] - 0.25 * step_m:
        grid = np.append(grid, s[-1])
    C = np.stack([np.interp(grid, s, Q[:, c]) for c in range(3)], 1)
    j = np.clip(np.searchsorted(s, grid), 1, len(s) - 1)
    Um = Uk[j - 1] | Uk[j]
    C = _gauss_smooth(C, step_m, smooth_m)
    C[:, 1] -= h_pivot_m  # от pivot к поверхности дороги
    return C, Um


def edges(C: np.ndarray, width_m: float):
    """Кромки по нормали к курсу в горизонтальной плоскости (Three: Y вверх). left = (fz, 0, -fx)."""
    t = np.gradient(C[:, [0, 2]], axis=0)
    nrm = np.linalg.norm(t, axis=1, keepdims=True)
    t = np.where(nrm > 1e-9, t / np.maximum(nrm, 1e-9), np.array([1.0, 0.0]))
    left_dir = np.stack([t[:, 1], np.zeros(len(C)), -t[:, 0]], 1)
    return C + left_dir * width_m / 2, C - left_dir * width_m / 2


def _r3(A) -> list:
    return [[round(float(v), 3) for v in p] for p in A]


def build_road(est_ue_cm, modes=None, fps: float = 30.0, **opt) -> dict | None:
    """Блок "road" для viewer.json по оценке траектории в Unreal (см). opt — ключи DEFAULT_ROAD."""
    o = {**DEFAULT_ROAD, **{k: v for k, v in opt.items() if k in DEFAULT_ROAD and v is not None}}
    P = np.array([[np.nan if v is None else v for v in p] for p in est_ue_cm], dtype=np.float64)
    res = centerline_from_track(_ue_to_three(P), modes, fps, h_pivot_m=o["h_pivot_m"], step_m=o["step_m"],
                                smooth_m=o["smooth_m"], max_predict_s=o["max_predict_s"])
    if res is None:
        return None
    C, unc = res
    width = float(o["width_m"] if o["width_m"] else o["truck_width_m"] * o["width_factor"])
    L, R = edges(C, width)
    length = float(np.hypot(np.diff(C[:, 0]), np.diff(C[:, 2])).sum())
    return {
        "units": "three_m", "step_m": o["step_m"], "width_m": round(width, 3),
        "h_pivot_m": o["h_pivot_m"], "length_m": round(length, 1), "n_points": int(len(C)),
        "centerline": _r3(C), "left": _r3(L), "right": _r3(R),
        "uncertain": _segments(unc),
        "ue_cm": {"centerline": [[round(float(v), 1) for v in p] for p in _three_to_ue(C)],
                  "left": [[round(float(v), 1) for v in p] for p in _three_to_ue(L)],
                  "right": [[round(float(v), 1) for v in p] for p in _three_to_ue(R)]},
        "source": "trajectory_smoothed",
        "note": ("Дорога = сглаженная траектория грузовика (он едет по центру). Система координат как у three_m: метры, "
                 "Y вверх. Высота = высота pivot − h_pivot_m (pivot выше земли). uncertain — отрезки индексов "
                 "centerline [i0, i1] (включительно), где оценка долго шла в режиме predict. Ширина — допущение."),
    }


def add_road(viewer: dict, obj_name: str | None = None, with_gt: bool = True, **opt) -> dict | None:
    """Дописать viewer["road"] (и road_gt, если у объекта есть эталон) по оценке объекта obj_name (по умолчанию первый)."""
    objs = viewer.get("objects", [])
    ob = next((o for o in objs if obj_name in (None, o["name"])), None)
    if ob is None:
        return None
    est = ob["estimate"]
    fps = viewer.get("fps", 30.0)
    road = build_road(est["ue_cm"], est.get("mode"), fps, **opt)
    if road is None:
        return None
    road["object"] = ob["name"]
    viewer["road"] = road
    viewer.pop("road_gt", None)
    if with_gt and ob.get("truth"):
        gt = build_road(ob["truth"]["ue_cm"], None, fps, **opt)
        if gt is not None:
            gt["source"] = "truth_smoothed"
            gt["object"] = ob["name"]
            viewer["road_gt"] = gt
    return road


def main(argv=None):
    ap = argparse.ArgumentParser(description="дописать road в готовый viewer.json")
    ap.add_argument("--viewer", required=True)
    ap.add_argument("--out", default=None, help="по умолчанию перезаписать --viewer")
    ap.add_argument("--object", default=None, help="имя объекта (по умолчанию первый; для грузовика — truck)")
    ap.add_argument("--width-m", type=float, default=None, help="ширина дороги, м (по умолчанию width_factor × ширина грузовика)")
    ap.add_argument("--width-factor", type=float, default=None)
    ap.add_argument("--step-m", type=float, default=None)
    ap.add_argument("--smooth-m", type=float, default=None)
    ap.add_argument("--no-gt", action="store_true", help="не писать road_gt")
    a = ap.parse_args(argv)
    v = json.loads(Path(a.viewer).read_text(encoding="utf-8"))
    road = add_road(v, a.object, not a.no_gt, width_m=a.width_m, width_factor=a.width_factor, step_m=a.step_m,
                    smooth_m=a.smooth_m)
    if road is None:
        raise SystemExit("не удалось построить дорогу (нет объекта или слишком мало точек)")
    p = Path(a.out or a.viewer)
    p.write_text(json.dumps(v, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{p}: road {road['n_points']} точек, {road['length_m']} м, ширина {road['width_m']} м, "
          f"uncertain {road['uncertain']}")


if __name__ == "__main__":
    main()
