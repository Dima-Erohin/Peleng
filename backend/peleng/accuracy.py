"""Оценка точности: неопределённость точки, проверка по доске, проверка по рулетке, отчёт.

* Неопределённость точки (pos_std_m): если каждая камера ошибается в положении точки на σ пикселей
  (шум детектора), то ковариация триангулированной точки ≈ σ² · (Σ Jᵢᵀ Jᵢ)⁻¹, где Jᵢ — якобиан
  проекции точки в камеру i (2×3, пиксели на метр). Якобиан берётся численно через CameraModel.project,
  так что дисторсия объектива учтена. Это оценка «из геометрии камер»: она не знает об ошибках калибровки,
  зато честно показывает, где камеры видят точку под плохим углом или издалека.
* Проверка по доске: углы ChArUco, найденные ≥ 2 камерами, триангулируются (как люди, а не через калибровку
  положения) → расстояния между соседними углами сравниваются с известным размером клетки.
* Проверка по рулетке: среднее положение человека/объекта в точках A и B → расстояние → сравнение с рулеткой.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .calibration import Board, BoardDetection
from .geometry import CameraModel, triangulate_robust

DEFAULT_SIGMA_PX = 2.0
MIN_PAIR_ANGLE_DEG = 20.0


# ---------------------------------------------------------------------------
# 1. Неопределённость точки
# ---------------------------------------------------------------------------

def projection_jacobian(cam: CameraModel, X: np.ndarray, h: float = 1e-4) -> np.ndarray | None:
    """Якобиан проекции d(пиксели)/d(X), (2,3). Численно (центральные разности) — с учётом дисторсии."""
    X = np.asarray(X, dtype=np.float64).reshape(3)
    if cam.depth(X)[0] <= 0.05:
        return None
    d = np.vstack([np.eye(3) * h, -np.eye(3) * h])
    px = cam.project(X + d)
    return ((px[:3] - px[3:]) / (2 * h)).T


def point_covariance(cams: list[CameraModel], X: np.ndarray, sigma_px: float = DEFAULT_SIGMA_PX) -> np.ndarray | None:
    """Ковариация (3,3) точки, триангулированной по камерам cams при шуме σ пикселей в каждой.
    None — меньше двух камер видят точку или геометрия вырождена (лучи почти параллельны)."""
    A = np.zeros((3, 3))
    n = 0
    for cam in cams:
        J = projection_jacobian(cam, X)
        if J is None:
            continue
        A += J.T @ J
        n += 1
    if n < 2:
        return None
    w = np.linalg.eigvalsh(A)
    if w[0] <= 1e-9 * max(w[-1], 1e-12):
        return None
    return sigma_px**2 * np.linalg.inv(A)


def pos_std(cams: list[CameraModel], X: np.ndarray, sigma_px: float = DEFAULT_SIGMA_PX) -> np.ndarray | None:
    """[σx, σy, σz] в метрах (1σ) или None."""
    C = point_covariance(cams, X, sigma_px)
    if C is None:
        return None
    return np.sqrt(np.maximum(np.diag(C), 0.0))


def ray_angle_deg(a: CameraModel, b: CameraModel, X: np.ndarray) -> float:
    """Угол между лучами из точки X в камеры a и b. 0° и 180° одинаково плохи (камеры на одной линии с точкой)."""
    X = np.asarray(X, dtype=np.float64).reshape(3)
    u = a.center - X
    v = b.center - X
    c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def effective_angle_deg(angle: float) -> float:
    """Угол, важный для триангуляции: камеры напротив друг друга (180°) так же плохи, как рядом (0°)."""
    return min(angle, 180.0 - angle)


# ---------------------------------------------------------------------------
# 2. Проверка по доске
# ---------------------------------------------------------------------------

def merge_detections(dets: list[BoardDetection], board: Board) -> BoardDetection | None:
    """Несколько кадров неподвижной доски с одной камеры → медиана положения каждого угла (меньше шума)."""
    dets = [d for d in dets if d is not None]
    if not dets:
        return None
    by_id: dict[int, list[np.ndarray]] = {}
    for d in dets:
        for i, c in zip(d.ids, d.corners):
            by_id.setdefault(int(i), []).append(c)
    ids = np.array(sorted(by_id), dtype=int)
    corners = np.array([np.median(np.array(by_id[i]), axis=0) for i in ids])
    return BoardDetection(corners, ids, board.all_corners[ids])


def neighbor_pairs(board: Board) -> list[tuple[int, int]]:
    """Пары соседних углов ChArUco (расстояние = одна клетка)."""
    P = board.all_corners
    out = []
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            if abs(np.linalg.norm(P[i] - P[j]) - board.square_m) < 1e-3 * board.square_m:
                out.append((i, j))
    return out


def board_check(
    board: Board,
    cams: dict[str, CameraModel],
    dets: dict[str, BoardDetection],
    reproj_thresh_px: float = 5.0,
    sigma_px: float = 0.5,
) -> dict:
    """Триангулировать углы доски по нескольким камерам и сравнить расстояния с размером клетки.

    cams и dets — по одной модели и одному (усреднённому) обнаружению доски на камеру, модель — под размер кадра.
    """
    views = {c: d for c, d in dets.items() if d is not None and c in cams and cams[c].calibrated}
    if len(views) < 2:
        return {"ok": False, "message": f"доска видна {len(views)} откалиброванным камерам (нужно ≥ 2)", "cams": sorted(views)}
    pts3d: dict[int, np.ndarray] = {}
    used_by: dict[int, list[str]] = {}
    reproj = []
    stds = []
    lookup = {c: {int(i): k for k, i in enumerate(d.ids)} for c, d in views.items()}
    for cid in range(len(board.all_corners)):
        cs = [c for c in views if cid in lookup[c]]
        if len(cs) < 2:
            continue
        pts = [views[c].corners[lookup[c][cid]] for c in cs]
        X, err, used = triangulate_robust([cams[c] for c in cs], pts, reproj_thresh_px=reproj_thresh_px)
        if X is None:
            continue
        pts3d[cid] = X
        used_by[cid] = [cs[u] for u in used]
        reproj.append(err)
        s = pos_std([cams[c] for c in used_by[cid]], X, sigma_px)
        if s is not None:
            stds.append(float(np.sqrt(np.mean(s**2))))  # σ вдоль произвольного направления
    if len(pts3d) < 4:
        return {"ok": False, "message": f"общих углов у камер: {len(pts3d)} (нужно ≥ 4)", "cams": sorted(views)}
    errs = []
    for i, j in neighbor_pairs(board):
        if i in pts3d and j in pts3d:
            errs.append(float(np.linalg.norm(pts3d[i] - pts3d[j]) - board.square_m))
    if not errs:
        return {"ok": False, "message": "нет соседних углов, видных ≥ 2 камерами", "cams": sorted(views)}
    errs_mm = np.array(errs) * 1000
    P = np.array(list(pts3d.values()))
    c = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - c)
    normal = Vt[-1]
    dev_mm = np.abs((P - c) @ normal) * 1000
    tilt = math.degrees(math.acos(min(1.0, abs(float(normal[2])))))
    # размах: самые далёкие друг от друга найденные углы (ошибка на длине всей доски)
    ids = list(pts3d)
    best = (0.0, 0.0)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            true_d = float(np.linalg.norm(board.all_corners[ids[a]] - board.all_corners[ids[b]]))
            if true_d > best[0]:
                best = (true_d, float(np.linalg.norm(pts3d[ids[a]] - pts3d[ids[b]])))
    cams_used = sorted({c for v in used_by.values() for c in v})
    return {
        "ok": True,
        "message": "ok",
        "cams": cams_used,
        "corners": len(pts3d),
        "pairs": len(errs),
        "square_mm": round(board.square_m * 1000, 2),
        "median_err_mm": round(float(np.median(np.abs(errs_mm))), 2),
        "max_err_mm": round(float(np.max(np.abs(errs_mm))), 2),
        "rms_err_mm": round(float(np.sqrt(np.mean(errs_mm**2))), 2),
        "bias_mm": round(float(np.mean(errs_mm)), 2),  # >0 — доска «больше» настоящей (масштаб)
        "median_err_pct": round(float(np.median(np.abs(errs_mm))) / (board.square_m * 1000) * 100, 2),
        "span_m": round(best[0], 4),
        "span_err_mm": round((best[1] - best[0]) * 1000, 2),
        "plane_rms_mm": round(float(np.sqrt(np.mean(dev_mm**2))), 2),
        "plane_max_mm": round(float(dev_mm.max()), 2),
        "tilt_deg": round(tilt, 1),
        "center": np.round(c, 3).tolist(),
        "reproj_px": round(float(np.mean(reproj)), 2),
        # ожидаемая ошибка расстояния между соседями при шуме углов sigma_px: √2 · σ точки
        "expected_err_mm": round(float(np.median(stds)) * math.sqrt(2) * 1000, 2) if stds else None,
        "sigma_px": sigma_px,
    }


# ---------------------------------------------------------------------------
# 3. Проверка по рулетке
# ---------------------------------------------------------------------------

def average_point(samples: list[dict]) -> dict | None:
    """samples: [{"t", "pos", "floor"?, "std"?, "mode"}] одного человека/объекта → среднее положение и разброс."""
    if not samples:
        return None
    P = np.array([s["pos"] for s in samples], dtype=np.float64)
    out = {
        "pos": np.round(P.mean(axis=0), 4).tolist(),
        "n": len(samples),
        "spread_m": round(float(np.sqrt(np.mean(np.sum((P - P.mean(axis=0)) ** 2, axis=1)))), 4),
        "tri_share": round(sum(s.get("mode") == "tri" for s in samples) / len(samples), 2),
    }
    F = [s["floor"] for s in samples if s.get("floor") is not None]
    if F:
        out["floor"] = np.round(np.mean(np.array(F, dtype=np.float64), axis=0), 4).tolist()
    S = [s["std"] for s in samples if s.get("std") is not None]
    if S:
        # σ среднего по n кадрам меньше, но шум соседних кадров не независим — берём σ одного кадра (честнее)
        out["std"] = np.round(np.median(np.array(S, dtype=np.float64), axis=0), 4).tolist()
    return out


def tape_result(a: dict, b: dict, measured_m: float, horizontal: bool = True) -> dict:
    """Сравнить расстояние A→B с измеренным рулеткой. horizontal — по полу (для людей: рулетка на полу)."""
    if horizontal and a.get("floor") is not None and b.get("floor") is not None:
        pa, pb = np.array(a["floor"], float), np.array(b["floor"], float)
    else:
        pa, pb = np.array(a["pos"], float), np.array(b["pos"], float)
        horizontal = False
    d = pb - pa
    dist = float(np.linalg.norm(d[:2])) if horizontal else float(np.linalg.norm(d))
    err = dist - float(measured_m)
    out = {
        "measured_m": round(float(measured_m), 4),
        "system_m": round(dist, 4),
        "err_cm": round(err * 100, 1),
        "err_pct": round(abs(err) / float(measured_m) * 100, 2) if measured_m > 0 else None,
        "horizontal": horizontal,
    }
    if a.get("std") is not None and b.get("std") is not None and dist > 1e-6:
        # σ расстояния = проекция σ обеих точек на направление A→B
        u = d / np.linalg.norm(d)
        if horizontal:
            u = np.array([d[0], d[1], 0.0]) / dist
        sa = float(np.sqrt(np.sum((np.array(a["std"]) * u) ** 2)))
        sb = float(np.sqrt(np.sum((np.array(b["std"]) * u) ** 2)))
        out["expected_cm"] = round(math.hypot(sa, sb) * 100, 1)
    return out


# ---------------------------------------------------------------------------
# История
# ---------------------------------------------------------------------------

def load_history(path: Path) -> list[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def append_history(path: Path, rec: dict) -> list[dict]:
    hist = load_history(path)
    hist.append({"time": round(time.time(), 1), **rec})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(hist, ensure_ascii=False, indent=1), encoding="utf-8")
    return hist


def summarize_history(hist: list[dict]) -> dict:
    tapes = [h for h in hist if h.get("type") == "tape"]
    boards = [h for h in hist if h.get("type") == "board" and h.get("ok")]
    out: dict = {"tape_checks": len(tapes), "board_checks": len(boards)}
    if tapes:
        e = np.abs([h["err_cm"] for h in tapes])
        out["tape_median_err_cm"] = round(float(np.median(e)), 1)
        out["tape_max_err_cm"] = round(float(np.max(e)), 1)
        pct = [h["err_pct"] for h in tapes if h.get("err_pct") is not None]
        if pct:
            out["tape_median_err_pct"] = round(float(np.median(pct)), 2)
    if boards:
        last = boards[-1]
        out["board_last"] = {k: last.get(k) for k in ("time", "median_err_mm", "max_err_mm", "plane_rms_mm", "cams", "corners")}
        out["board_median_err_mm"] = round(float(np.median([b["median_err_mm"] for b in boards])), 2)
    return out


# ---------------------------------------------------------------------------
# 4. Отчёт по камерам
# ---------------------------------------------------------------------------

def camera_report(cams: dict[str, CameraModel], center: np.ndarray, sigma_px: float = DEFAULT_SIGMA_PX) -> dict:
    """Ошибки калибровки по камерам, углы между лучами пар камер в центре зоны, σ в центре зоны.

    Модели камер — под разрешение, в котором работает трекинг (σ задана в его пикселях)."""
    center = np.asarray(center, dtype=np.float64).reshape(3)
    per_cam = []
    warnings = []
    calibrated = {c: m for c, m in cams.items() if m.calibrated}
    for cid, m in cams.items():
        d = {
            "id": cid,
            "calibrated": m.calibrated,
            "intrinsics": m.meta.get("intrinsics", "approx"),
            "intrinsics_rms": m.meta.get("intrinsics_rms"),
            "extrinsics_reproj_px": m.meta.get("extrinsics_reproj_px"),
            "extrinsics_points": m.meta.get("extrinsics_points"),
            "hfov_deg": round(m.hfov_deg(), 1),
        }
        if m.calibrated:
            d["center"] = np.round(m.center, 3).tolist()
            d["dist_to_zone_m"] = round(float(np.linalg.norm(m.center - center)), 2)
            in_front = m.depth(center)[0] > 0.05
            px = m.project(center)[0] if in_front else None
            d["sees_zone_center"] = bool(in_front and 0 <= px[0] < m.size[0] and 0 <= px[1] < m.size[1])
            if not d["sees_zone_center"]:
                warnings.append(f"{cid}: центр зоны вне кадра")
            r = d["extrinsics_reproj_px"]
            if r is not None and r > 1.5:
                warnings.append(f"{cid}: ошибка калибровки положения {r} px (> 1.5) — перекалибровать")
            if d["intrinsics"] in ("approx", "approx+focal"):
                warnings.append(f"{cid}: оптика приблизительная (по углу обзора) — лучше «Оптика: 15 кадров»")
        else:
            warnings.append(f"{cid}: положение не откалибровано")
        per_cam.append(d)
    pairs = []
    ids = sorted(calibrated)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ang = ray_angle_deg(calibrated[a], calibrated[b], center)
            eff = effective_angle_deg(ang)
            s = pos_std([calibrated[a], calibrated[b]], center, sigma_px)
            p = {"a": a, "b": b, "angle_deg": round(ang, 1), "effective_deg": round(eff, 1), "ok": eff >= MIN_PAIR_ANGLE_DEG}
            if s is not None:
                p["sigma_cm"] = np.round(s * 100, 1).tolist()
            if not p["ok"]:
                warnings.append(f"{a}–{b}: угол между лучами {eff:.0f}° (< {MIN_PAIR_ANGLE_DEG:.0f}°) — камеры почти на одной линии")
            pairs.append(p)
    out = {"zone_center": np.round(center, 3).tolist(), "sigma_px": sigma_px, "cameras": per_cam, "pairs": pairs, "warnings": warnings}
    s = pos_std(list(calibrated.values()), center, sigma_px)
    if s is not None:
        out["zone_sigma_cm"] = np.round(s * 100, 1).tolist()
        out["zone_sigma_h_cm"] = round(float(math.hypot(s[0], s[1])) * 100, 1)
    return out
