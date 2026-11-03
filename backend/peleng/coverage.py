"""Карта покрытия: какие участки пола видят ≥ 2 / 1 / 0 камер и насколько хорошо там работает триангуляция.

Помогает расставлять камеры: откалибровали положение → посмотрели карту → передвинули телефон → перекалибровали.

* Сетка — на горизонтальной плоскости z = height_m (по умолчанию 0.9 м — высота таза: именно эта точка человека
  триангулируется и попадает в pos). height_m = 0 — покрытие самого пола.
* Точка X «видна» камерой, если:
  глубина (координата z в системе камеры) > min_depth_m, расстояние от камеры ≤ max_range_m
  (дальше человек слишком мелкий для детектора) и проекция X попадает внутрь кадра с отступом margin_frac
  от края (у края кадра человек обычно обрезан, и скелет не находится).
* Качество (для клеток с ≥ 2 камерами) — ожидаемая ошибка положения по горизонтали σ_h (1σ, см) при шуме детекции
  sigma_px пикселей: та же формула, что accuracy.point_covariance (σ² · (Σ Jᵢᵀ Jᵢ)⁻¹), только для всей сетки сразу
  и с якобианом камеры-обскуры (без дисторсии). Плюс лучший «эффективный» угол между лучами пары камер
  (accuracy.effective_angle_deg: 0° и 180° одинаково плохи). Число quality ∈ [0, 1] — σ_h в логарифмической шкале:
  1 при σ_h ≤ good_cm, 0 при σ_h ≥ bad_cm.
* σ задаётся в пикселях кадра трекинга (pipeline.track_width, у телефонов 640): модели камер пересчитываются под эту ширину.
  Видимость от разрешения не зависит (проверяется в долях кадра), поэтому годится модель под любое разрешение.

Подробно — docs/coverage.md.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .geometry import CameraModel

DEFAULTS = {
    "cell_m": 0.2,  # шаг сетки, м
    "height_m": 0.9,  # высота плоскости, на которой считается покрытие (таз человека)
    "min_depth_m": 0.2,  # точка ближе к камере (по оси взгляда) — не видна
    "max_range_m": 10.0,  # дальше — человек слишком мелкий для детектора
    "margin_frac": 0.03,  # отступ от края кадра, доля ширины/высоты
    "sigma_px": 2.0,  # шум положения точки в кадре трекинга, пиксели (как accuracy.sigma_px)
    "ref_width_px": None,  # ширина кадра, в пикселях которого sigma_px; по умолчанию pipeline.track_width
    "good_cm": 1.0,  # σ_h, при которой quality = 1
    "bad_cm": 10.0,  # σ_h, при которой quality = 0
    "max_extent_m": 12.0,  # сетка не больше max_extent_m × max_extent_m
    "max_cells": 40000,  # если клеток больше — шаг увеличивается
    "pad_m": 0.3,  # запас вокруг области, видимой камерами (если размер комнаты не задан)
}

MIN_CELL_M = 0.05
MAX_CELL_M = 1.0


# ---------------------------------------------------------------------------
# Видимость и качество в наборе точек
# ---------------------------------------------------------------------------

def visible_mask(cam: CameraModel, X: np.ndarray, min_depth_m: float = 0.2, max_range_m: float = 10.0,
                 margin_frac: float = 0.03) -> np.ndarray:
    """(N,) bool — видит ли камера точки X (N,3). Определение — в описании модуля."""
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    Xc = X @ cam.R.T + cam.t.reshape(3)
    z = Xc[:, 2]
    ok = (z > min_depth_m) & (np.linalg.norm(X - cam.center, axis=1) <= max_range_m)
    w, h = cam.size
    mx, my = margin_frac * w, margin_frac * h
    K = cam.K
    zs = np.where(ok, z, 1.0)
    u = K[0, 0] * Xc[:, 0] / zs + K[0, 1] * Xc[:, 1] / zs + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / zs + K[1, 2]
    # сначала — без дисторсии и с запасом: отсекает точки далеко за кадром, на которых полином дисторсии «заворачивается»
    loose = ok & (u > -0.5 * w) & (u < 1.5 * w) & (v > -0.5 * h) & (v < 1.5 * h)
    if np.any(cam.dist) and loose.any():
        px = cam.project(X[loose])
        u = u.copy()
        v = v.copy()
        u[loose], v[loose] = px[:, 0], px[:, 1]
    return loose & (u >= mx) & (u <= w - mx) & (v >= my) & (v <= h - my)


def pinhole_jacobians(cam: CameraModel, X: np.ndarray) -> np.ndarray:
    """Якобианы проекции d(пиксели)/d(X) для точек X (N,3) → (N,2,3), модель камеры-обскуры (без дисторсии)."""
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    Xc = X @ cam.R.T + cam.t.reshape(3)
    z = np.where(np.abs(Xc[:, 2]) < 1e-9, 1e-9, Xc[:, 2])
    x, y = Xc[:, 0] / z, Xc[:, 1] / z
    R = cam.R
    du = (R[0][None, :] - x[:, None] * R[2][None, :]) / z[:, None]  # d(x)/dX
    dv = (R[1][None, :] - y[:, None] * R[2][None, :]) / z[:, None]  # d(y)/dX
    K = cam.K
    Ju = K[0, 0] * du + K[0, 1] * dv
    Jv = K[1, 1] * dv
    return np.stack([Ju, Jv], axis=1)


def _ref_model(cam: CameraModel, ref_width_px: float | None) -> CameraModel:
    """Модель под ширину кадра трекинга (σ в его пикселях). Камеры с кадром уже ref_width_px не растягиваются."""
    if not ref_width_px or cam.size[0] <= ref_width_px:
        return cam
    w = int(round(ref_width_px))
    h = int(round(cam.size[1] * w / cam.size[0]))
    return cam.resized((w, h))


def evaluate_points(cams: dict[str, CameraModel], X: np.ndarray, params: dict | None = None) -> dict:
    """Покрытие набора точек X (N,3) откалиброванными камерами cams.

    Возвращает numpy-массивы длины N: counts (сколько камер видят), sigma_h_cm (NaN, если < 2 камер или геометрия вырождена),
    angle_deg (лучший эффективный угол между лучами пары камер; NaN при < 2), quality (0..1; NaN при < 2),
    и visible: {id камеры: (N,) bool}.
    """
    p = {**DEFAULTS, **(params or {})}
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    n = len(X)
    cams = {c: m for c, m in cams.items() if m is not None and m.calibrated}
    vis = {c: visible_mask(m, X, float(p["min_depth_m"]), float(p["max_range_m"]), float(p["margin_frac"])) for c, m in cams.items()}
    counts = np.zeros(n, dtype=int)
    A = np.zeros((n, 3, 3))
    for c, m in cams.items():
        v = vis[c]
        counts += v
        if v.any():
            J = pinhole_jacobians(_ref_model(m, p["ref_width_px"]), X[v])
            A[v] += np.einsum("nki,nkj->nij", J, J)
    sigma_h = np.full(n, np.nan)
    multi = counts >= 2
    if multi.any():
        Am = A[multi]
        w = np.linalg.eigvalsh(Am)
        good = w[:, 0] > 1e-9 * np.maximum(w[:, -1], 1e-12)
        cov = np.full((len(Am), 3, 3), np.nan)
        if good.any():
            cov[good] = float(p["sigma_px"]) ** 2 * np.linalg.inv(Am[good])
        sh = np.sqrt(np.maximum(cov[:, 0, 0] + cov[:, 1, 1], 0.0)) * 100.0
        sigma_h[multi] = sh
    # лучший эффективный угол между лучами пары камер, видящих точку
    angle = np.full(n, np.nan)
    ids = sorted(cams)
    dirs = {}
    for c in ids:
        d = cams[c].center[None, :] - X
        dirs[c] = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            both = vis[a] & vis[b]
            if not both.any():
                continue
            cosv = np.clip(np.sum(dirs[a][both] * dirs[b][both], axis=1), -1.0, 1.0)
            ang = np.degrees(np.arccos(cosv))
            eff = np.minimum(ang, 180.0 - ang)  # как accuracy.effective_angle_deg
            cur = angle[both]
            angle[both] = np.where(np.isnan(cur), eff, np.fmax(cur, eff))
    good_cm, bad_cm = float(p["good_cm"]), float(p["bad_cm"])
    q = np.full(n, np.nan)
    q[multi] = 0.0
    ok = np.isfinite(sigma_h)
    q[ok] = np.clip(np.log(bad_cm / np.maximum(sigma_h[ok], 1e-9)) / math.log(bad_cm / good_cm), 0.0, 1.0)
    return {"counts": counts, "sigma_h_cm": sigma_h, "angle_deg": angle, "quality": q, "visible": vis}



# ---------------------------------------------------------------------------
# Границы сетки
# ---------------------------------------------------------------------------

def _floor_footprint(cam: CameraModel, z0: float, max_range_m: float) -> np.ndarray:
    """Точки (M,2): где лучи из кадра (сетка 7×7) пересекают плоскость z = z0, не дальше max_range_m по горизонтали."""
    w, h = cam.size
    us, vs = np.meshgrid(np.linspace(0, w, 7), np.linspace(0, h, 7))
    px = np.stack([us.ravel(), vs.ravel()], axis=1)
    n = cv2.undistortPoints(px.reshape(-1, 1, 2), cam.K, cam.dist).reshape(-1, 2) if np.any(cam.dist) else (px - cam.K[:2, 2]) / np.diag(cam.K)[:2]
    d = np.hstack([n, np.ones((len(n), 1))]) @ cam.R  # направления в мире
    C = cam.center
    out = [C[:2]]
    for di in d:
        s = (z0 - C[2]) / di[2] if abs(di[2]) > 1e-9 else -1.0
        hd = di[:2] / max(np.linalg.norm(di[:2]), 1e-9)
        if s > 0:
            P = C[:2] + s * di[:2]
            r = np.linalg.norm(P - C[:2])
            out.append(P if r <= max_range_m else C[:2] + hd * max_range_m)
        elif np.linalg.norm(di[:2]) > 1e-6:  # луч уходит вверх (над горизонтом) — до дальности
            out.append(C[:2] + hd * max_range_m)
    return np.array(out)


def grid_bounds(cams: dict[str, CameraModel], room: dict | None, params: dict | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """(min_xy, max_xy, источник): по комнате (room.size_m, room.board_offset_m), если она задана, иначе — по тому,
    куда смотрят камеры, с запасом pad_m. Размер ограничен max_extent_m (обрезается вокруг центра)."""
    p = {**DEFAULTS, **(params or {})}
    room = room or {}
    if room.get("size_m"):
        w, d = float(room["size_m"][0]), float(room["size_m"][1])
        off = room.get("board_offset_m") or [0.0, 0.0]
        c = np.array([-float(off[0]), -float(off[1])])  # центр комнаты (как коробка комнаты во вьювере)
        lo, hi, src = c - [w / 2, d / 2], c + [w / 2, d / 2], "room"
    else:
        pts = [_floor_footprint(m, float(p["height_m"]), float(p["max_range_m"])) for m in cams.values() if m.calibrated]
        if not pts:
            return np.array([-2.0, -2.0]), np.array([2.0, 2.0]), "default"
        P = np.vstack(pts)
        lo, hi, src = P.min(axis=0) - p["pad_m"], P.max(axis=0) + p["pad_m"], "cameras"
    ext = float(p["max_extent_m"])
    c = (lo + hi) / 2
    half = np.minimum((hi - lo) / 2, ext / 2)
    return c - half, c + half, src


# ---------------------------------------------------------------------------
# Карта покрытия
# ---------------------------------------------------------------------------

def _r(a: np.ndarray, nd: int) -> list:
    """Массив → список для JSON, NaN → None."""
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a]


def coverage_map(cams: dict[str, CameraModel], room: dict | None = None, params: dict | None = None) -> dict:
    """Карта покрытия по откалиброванным камерам. Формат ответа — docs/coverage.md."""
    p = {**DEFAULTS, **{k: v for k, v in (params or {}).items() if v is not None}}
    cams = {c: m for c, m in cams.items() if m is not None and m.calibrated}
    lo, hi, src = grid_bounds(cams, room, p)
    cell = float(min(max(float(p["cell_m"]), MIN_CELL_M), MAX_CELL_M))
    size = hi - lo
    while math.ceil(size[0] / cell - 1e-9) * math.ceil(size[1] / cell - 1e-9) > int(p["max_cells"]):
        cell *= 1.25
    nx = max(1, math.ceil(size[0] / cell - 1e-9))
    ny = max(1, math.ceil(size[1] / cell - 1e-9))
    origin = (lo + hi) / 2 - np.array([nx, ny]) * cell / 2  # сетка по центру области
    z0 = float(p["height_m"])
    xs = origin[0] + (np.arange(nx) + 0.5) * cell
    ys = origin[1] + (np.arange(ny) + 0.5) * cell
    gx, gy = np.meshgrid(xs, ys)  # строки — по y: индекс клетки iy * nx + ix
    X = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z0)], axis=1)
    ev = evaluate_points(cams, X, p)
    counts = ev["counts"]
    area = cell * cell
    multi = counts >= 2
    sig = ev["sigma_h_cm"][multi]
    sig = sig[np.isfinite(sig)]
    summary = {
        "area_m2": round(float(counts.size * area), 2),
        "area_2plus_m2": round(float(np.sum(multi) * area), 2),
        "area_1_m2": round(float(np.sum(counts == 1) * area), 2),
        "area_0_m2": round(float(np.sum(counts == 0) * area), 2),
        "pct_2plus": round(100.0 * float(np.mean(multi)), 1),
        "sigma_h_cm_median": round(float(np.median(sig)), 2) if len(sig) else None,
        "sigma_h_cm_p90": round(float(np.percentile(sig, 90)), 2) if len(sig) else None,
        "area_good_m2": round(float(np.sum(ev["quality"][multi] >= 0.5) * area), 2),  # σ_h не хуже √(good·bad) (≈ 3 см)
    }
    cam_list = []
    for c in sorted(cams):
        m = cams[c]
        cam_list.append({"id": c, "center": np.round(m.center, 3).tolist(), "size": list(m.size), "hfov_deg": round(m.hfov_deg(), 1),
                         "area_m2": round(float(np.sum(ev["visible"][c]) * area), 2)})
    warnings = []
    if len(cams) == 0:
        warnings.append("нет откалиброванных камер — откалибруйте положение по доске")
    elif len(cams) == 1:
        warnings.append("откалибрована одна камера — 3D по двум камерам невозможно, только «ноги на полу»")
    elif summary["area_2plus_m2"] == 0:
        warnings.append("нет участков, которые видят ≥ 2 камеры, — поверните камеры на общую зону")
    for cm in cam_list:
        if cm["area_m2"] == 0:
            warnings.append(f"{cm['id']}: не видит ни одной клетки на высоте {z0:g} м")
    return {
        "ok": len(cams) > 0,
        "origin": np.round(origin, 4).tolist(),  # [x, y] угла клетки (0, 0) с наименьшими x, y
        "cell_m": round(cell, 4),
        "nx": nx,
        "ny": ny,
        "height_m": z0,
        "bounds_from": src,  # room | cameras | default
        "counts": counts.astype(int).tolist(),
        "quality": _r(ev["quality"], 3),
        "sigma_h_cm": _r(ev["sigma_h_cm"], 2),
        "angle_deg": _r(ev["angle_deg"], 1),
        "summary": summary,
        "cameras": cam_list,
        "warnings": warnings,
        "params": {k: p[k] for k in ("min_depth_m", "max_range_m", "margin_frac", "sigma_px", "ref_width_px", "good_cm", "bad_cm")},
    }
