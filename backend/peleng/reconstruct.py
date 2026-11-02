"""Сопоставление людей между камерами и восстановление 3D.

* ≥ 2 камер видят человека -> триангуляция всех 17 точек скелета ("tri").
* 1 камера -> допущение «ноги на полу»: луч через середину стоп пересекается с z = 0 ("mono").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import CameraModel, essential_between, ray_plane_z, symmetric_epipolar_px, triangulate_robust
from .pose import L_ANKLE, L_HIP, R_ANKLE, R_HIP, Person2D

DetKey = tuple[str, int]  # (camera_id, индекс человека в этой камере)


@dataclass
class Observation3D:
    pos: np.ndarray  # (3,) центр таза (или оценка)
    floor: np.ndarray  # (3,) точка на полу под человеком
    mode: str  # "tri" | "mono"
    cams: list[str]
    reproj_px: float = float("nan")
    kp3d: np.ndarray | None = None  # (17,3), NaN где нет
    dets: dict[str, int] = field(default_factory=dict)


def associate(
    cams: dict[str, CameraModel],
    dets: dict[str, list[Person2D]],
    kp_conf: float = 0.35,
    thresh_px: float = 25.0,
    min_common: int = 3,
) -> list[dict[str, int]]:
    """Группирует детекции разных камер, принадлежащие одному человеку.

    Стоимость пары = медиана симметричного эпиполярного расстояния по общим уверенным точкам.
    Жадная кластеризация: в кластере не больше одной детекции с каждой камеры.
    """
    ids = [c for c in dets if c in cams and cams[c].calibrated]
    norm: dict[DetKey, np.ndarray] = {}
    for c in ids:
        for i, p in enumerate(dets[c]):
            norm[(c, i)] = cams[c].undistort(p.kp)

    edges: list[tuple[float, DetKey, DetKey]] = []
    for ai, a in enumerate(ids):
        for b in ids[ai + 1:]:
            E = essential_between(cams[a], cams[b])
            for i, pa in enumerate(dets[a]):
                for j, pb in enumerate(dets[b]):
                    m = (pa.kc > kp_conf) & (pb.kc > kp_conf)
                    if m.sum() < min_common:
                        continue
                    d = symmetric_epipolar_px(E, norm[(a, i)][m], norm[(b, j)][m], cams[a].focal, cams[b].focal)
                    cost = float(np.median(d))
                    if cost < thresh_px:
                        edges.append((cost, (a, i), (b, j)))
    edges.sort(key=lambda e: e[0])

    cluster_of: dict[DetKey, int] = {}
    clusters: list[dict[str, int]] = []
    for c in ids:
        for i in range(len(dets[c])):
            cluster_of[(c, i)] = len(clusters)
            clusters.append({c: i})
    for _, u, v in edges:
        cu, cv = cluster_of[u], cluster_of[v]
        if cu == cv:
            continue
        A, B = clusters[cu], clusters[cv]
        if set(A) & set(B):
            continue  # нельзя: две детекции с одной камеры
        A.update(B)
        for c, i in B.items():
            cluster_of[(c, i)] = cu
        clusters[cv] = {}
    return [c for c in clusters if c]


def triangulate_cluster(
    cams: dict[str, CameraModel],
    dets: dict[str, list[Person2D]],
    cluster: dict[str, int],
    kp_conf: float = 0.35,
    reproj_thresh_px: float = 15.0,
) -> Observation3D | None:
    kp3d = np.full((17, 3), np.nan)
    errs = []
    used: set[str] = set()
    for k in range(17):
        cs, pts, ws = [], [], []
        for c, i in cluster.items():
            p = dets[c][i]
            if p.kc[k] > kp_conf:
                cs.append(c)
                pts.append(p.kp[k])
                ws.append(p.kc[k])
        if len(cs) < 2:
            continue
        X, err, used_idx = triangulate_robust([cams[c] for c in cs], pts, ws, reproj_thresh_px)
        if X is None:
            continue
        kp3d[k] = X
        errs.append(err)
        used.update(cs[u] for u in used_idx)
    valid = ~np.isnan(kp3d[:, 0])
    if valid.sum() < 4:
        return None
    hips = [k for k in (L_HIP, R_HIP) if valid[k]]
    pos = kp3d[hips].mean(axis=0) if hips else np.nanmean(kp3d[valid], axis=0)
    ankles = [k for k in (L_ANKLE, R_ANKLE) if valid[k]]
    if ankles:
        a = kp3d[ankles].mean(axis=0)
        floor = np.array([a[0], a[1], 0.0])
    else:
        floor = np.array([pos[0], pos[1], 0.0])
    if not (-1.0 < pos[2] < 3.0):  # явно мусор
        return None
    return Observation3D(pos, floor, "tri", sorted(used), float(np.mean(errs)), kp3d, dict(cluster))


ANKLE_HEIGHT_M = 0.08  # точка «лодыжка» у YOLO примерно на 8 см выше пола


def mono_observation(cam_id: str, cam: CameraModel, p: Person2D, kp_conf: float = 0.35, hip_height: float = 0.9) -> Observation3D | None:
    """Одна камера: точка на полу по стопам (или по низу рамки, если стопы не видны)."""
    w, h = cam.size
    if p.kc[L_ANKLE] > kp_conf or p.kc[R_ANKLE] > kp_conf:
        sel = [k for k in (L_ANKLE, R_ANKLE) if p.kc[k] > kp_conf]
        foot = p.kp[sel].mean(axis=0)
        z_plane = ANKLE_HEIGHT_M
    else:
        x1, y1, x2, y2 = p.bbox
        if y2 > h - 3:  # низ рамки обрезан краем кадра — ноги не видны, оценка неверна
            return None
        foot = np.array([(x1 + x2) / 2, y2])
        z_plane = 0.0
    o, d = cam.rays(foot.reshape(1, 2))
    X = ray_plane_z(o, d[0], z_plane)
    if X is None or np.linalg.norm(X - o) > 30:
        return None
    floor = np.array([X[0], X[1], 0.0])
    return Observation3D(floor + np.array([0, 0, hip_height]), floor, "mono", [cam_id], dets={cam_id: 0})


def reconstruct(
    cams: dict[str, CameraModel],
    dets: dict[str, list[Person2D]],
    kp_conf: float = 0.35,
    epi_thresh_px: float = 25.0,
    reproj_thresh_px: float = 15.0,
    mono_fallback: bool = True,
) -> list[Observation3D]:
    clusters = associate(cams, dets, kp_conf, epi_thresh_px)
    obs: list[Observation3D] = []
    singles: list[tuple[str, int]] = []
    for cl in clusters:
        if len(cl) >= 2:
            o = triangulate_cluster(cams, dets, cl, kp_conf, reproj_thresh_px)
            if o is not None:
                obs.append(o)
                continue
            # не сошлось — пусть каждая детекция попробует моно-режим
            singles.extend(cl.items())
        else:
            singles.extend(cl.items())
    if mono_fallback:
        for c, i in singles:
            o = mono_observation(c, cams[c], dets[c][i], kp_conf)
            if o is None:
                continue
            o.dets = {c: i}
            # не дублировать человека, уже найденного триангуляцией
            if any(np.linalg.norm(o.floor[:2] - t.floor[:2]) < 0.6 for t in obs if t.mode == "tri"):
                continue
            obs.append(o)
    return obs
