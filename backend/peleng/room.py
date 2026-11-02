"""Грубая 3D-«комната точками» по откалиброванным камерам.

Ищем одинаковые особые точки (SIFT) на кадрах пар камер, оставляем только пары,
согласованные с известной геометрией (эпиполярное ограничение), и триангулируем.
Чем ближе камеры друг к другу по направлению взгляда, тем больше точек.
Камеры «лицом друг к другу» видят разные стороны предметов — точек будет мало (это нормально).
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import CameraModel, essential_between, symmetric_epipolar_px


def scan_points(
    cams: dict[str, CameraModel],
    images: dict[str, np.ndarray],
    epi_thresh_px: float = 3.0,
    reproj_thresh_px: float = 4.0,
    max_points: int = 30000,
    max_range_m: float = 20.0,
) -> dict:
    ids = [c for c in images if c in cams and cams[c].calibrated]
    sift = cv2.SIFT_create(nfeatures=6000)
    feats = {}
    for c in ids:
        gray = cv2.cvtColor(images[c], cv2.COLOR_BGR2GRAY)
        kps, desc = sift.detectAndCompute(gray, None)
        if desc is None:
            continue
        pts = np.array([k.pt for k in kps], dtype=np.float64)
        feats[c] = (pts, desc)

    all_X, all_C, pairs = [], [], []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    keys = [c for c in ids if c in feats]
    for ai, a in enumerate(keys):
        for b in keys[ai + 1:]:
            pa, da = feats[a]
            pb, db = feats[b]
            if len(pa) < 10 or len(pb) < 10:
                continue
            knn = matcher.knnMatch(da, db, k=2)
            good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
            if len(good) < 8:
                pairs.append({"a": a, "b": b, "matches": 0, "points": 0})
                continue
            ia = np.array([m.queryIdx for m in good])
            ib = np.array([m.trainIdx for m in good])
            A, B = cams[a], cams[b]
            na, nb = A.undistort(pa[ia]), B.undistort(pb[ib])
            d = symmetric_epipolar_px(essential_between(A, B), na, nb, A.focal, B.focal)
            keep = d < epi_thresh_px
            na, nb, ia, ib = na[keep], nb[keep], ia[keep], ib[keep]
            if len(na) == 0:
                pairs.append({"a": a, "b": b, "matches": len(good), "points": 0})
                continue
            Xh = cv2.triangulatePoints(A.P_norm, B.P_norm, na.T, nb.T)
            X = (Xh[:3] / Xh[3]).T
            ok = (A.depth(X) > 0.1) & (B.depth(X) > 0.1) & (np.linalg.norm(X, axis=1) < max_range_m)
            ea = np.linalg.norm(A.project(X) - pa[ia], axis=1)
            eb = np.linalg.norm(B.project(X) - pb[ib], axis=1)
            ok &= (ea < reproj_thresh_px) & (eb < reproj_thresh_px)
            X = X[ok]
            px = pa[ia][ok].round().astype(int)
            img = images[a]
            px[:, 0] = px[:, 0].clip(0, img.shape[1] - 1)
            px[:, 1] = px[:, 1].clip(0, img.shape[0] - 1)
            col = img[px[:, 1], px[:, 0]][:, ::-1]  # BGR -> RGB
            all_X.append(X)
            all_C.append(col)
            pairs.append({"a": a, "b": b, "matches": len(good), "points": int(len(X))})

    if all_X:
        X = np.vstack(all_X)
        C = np.vstack(all_C)
    else:
        X = np.zeros((0, 3))
        C = np.zeros((0, 3), dtype=np.uint8)
    if len(X) > max_points:
        sel = np.random.default_rng(0).choice(len(X), max_points, replace=False)
        X, C = X[sel], C[sel]
    return {
        "type": "room_points",
        "points": np.round(X, 3).tolist(),
        "colors": C.astype(int).tolist(),
        "pairs": pairs,
    }
