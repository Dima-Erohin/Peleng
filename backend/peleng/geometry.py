"""Модель камеры и базовая многовидовая геометрия.

Система координат мира: метры, Z вверх, пол — плоскость z = 0.
Внешние параметры камеры (R, t) переводят точку мира в систему камеры:
    X_cam = R @ X_world + t
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


def approx_K(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Приблизительная матрица камеры по горизонтальному углу обзора.

    Для основного (1x) объектива телефона hfov ≈ 65–72°, для вебкамеры ноутбука ≈ 60–70°.
    """
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], dtype=np.float64)


def scale_K(K: np.ndarray, src_size: tuple[int, int], dst_size: tuple[int, int]) -> np.ndarray:
    """Пересчитать K под другое разрешение (при том же соотношении сторон)."""
    sx = dst_size[0] / src_size[0]
    sy = dst_size[1] / src_size[1]
    K2 = K.copy()
    K2[0, :] *= sx
    K2[1, :] *= sy
    K2[2, 2] = 1.0
    return K2


@dataclass
class CameraModel:
    K: np.ndarray
    dist: np.ndarray
    size: tuple[int, int]  # (width, height)
    R: np.ndarray | None = None
    t: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def calibrated(self) -> bool:
        return self.R is not None and self.t is not None

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def P_norm(self) -> np.ndarray:
        """Матрица проекции в нормализованных (неискажённых) координатах: [R|t]."""
        return np.hstack([self.R, self.t.reshape(3, 1)])

    @property
    def focal(self) -> float:
        return float((self.K[0, 0] + self.K[1, 1]) / 2.0)

    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.size[0] / 2.0 / self.K[0, 0]))

    def undistort(self, pts: np.ndarray) -> np.ndarray:
        """Пиксели (N,2) -> нормализованные координаты (N,2) без дисторсии."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.zeros((0, 2))
        return cv2.undistortPoints(pts, self.K, self.dist).reshape(-1, 2)

    def project(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        if len(X) == 0:
            return np.zeros((0, 2))
        rvec, _ = cv2.Rodrigues(self.R)
        px, _ = cv2.projectPoints(X, rvec, self.t.reshape(3, 1), self.K, self.dist)
        return px.reshape(-1, 2)

    def depth(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        return (X @ self.R.T + self.t)[:, 2]

    def rays(self, pts_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Лучи в мировой системе: (origin(3,), directions(N,3) единичные)."""
        n = self.undistort(pts_px)
        d_cam = np.hstack([n, np.ones((len(n), 1))])
        d = d_cam @ self.R  # == (R.T @ d_cam.T).T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return self.center, d

    # --- сериализация ---
    def to_dict(self) -> dict:
        d = {
            "K": self.K.tolist(),
            "dist": self.dist.reshape(-1).tolist(),
            "size": list(self.size),
            "meta": self.meta,
        }
        if self.calibrated:
            d["R"] = self.R.tolist()
            d["t"] = self.t.reshape(-1).tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CameraModel":
        R = np.array(d["R"], dtype=np.float64) if "R" in d else None
        t = np.array(d["t"], dtype=np.float64) if "t" in d else None
        return cls(
            K=np.array(d["K"], dtype=np.float64),
            dist=np.array(d["dist"], dtype=np.float64),
            size=tuple(d["size"]),
            R=R,
            t=t,
            meta=d.get("meta", {}),
        )

    def resized(self, size: tuple[int, int]) -> "CameraModel":
        if tuple(size) == tuple(self.size):
            return self
        return CameraModel(scale_K(self.K, self.size, size), self.dist.copy(), tuple(size), self.R, self.t, dict(self.meta))


# ---------------------------------------------------------------------------
# Триангуляция
# ---------------------------------------------------------------------------

def triangulate_dlt(P_list: list[np.ndarray], xy_list: list[np.ndarray], w_list: list[float] | None = None) -> np.ndarray:
    """Взвешенная DLT-триангуляция одной точки по N видам (нормализованные координаты)."""
    rows = []
    for i, (P, xy) in enumerate(zip(P_list, xy_list)):
        w = 1.0 if w_list is None else float(w_list[i])
        rows.append(w * (xy[0] * P[2] - P[0]))
        rows.append(w * (xy[1] * P[2] - P[1]))
    A = np.asarray(rows)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def triangulate_robust(
    cams: list[CameraModel],
    pts_px: list[np.ndarray],
    weights: list[float] | None = None,
    reproj_thresh_px: float = 15.0,
) -> tuple[np.ndarray | None, float, list[int]]:
    """Триангуляция с отбрасыванием худшего вида, пока ошибка репроекции велика.

    Возвращает (X или None, средняя ошибка репроекции в пикселях, индексы использованных видов).
    """
    idx = list(range(len(cams)))
    w_all = [1.0] * len(cams) if weights is None else list(weights)
    norm = [cams[i].undistort(np.asarray(pts_px[i]).reshape(1, 2))[0] for i in idx]
    while len(idx) >= 2:
        X = triangulate_dlt([cams[i].P_norm for i in idx], [norm[i] for i in idx], [w_all[i] for i in idx])
        errs = []
        ok = True
        for i in idx:
            if cams[i].depth(X)[0] <= 0.05:
                ok = False
                errs.append(1e9)
                continue
            errs.append(float(np.linalg.norm(cams[i].project(X)[0] - np.asarray(pts_px[i]).reshape(2))))
        errs_arr = np.array(errs)
        if ok and errs_arr.max() <= reproj_thresh_px:
            return X, float(errs_arr.mean()), idx
        if len(idx) == 2:
            break
        idx.pop(int(np.argmax(errs_arr)))
    return None, float("inf"), []


def ray_plane_z(origin: np.ndarray, direction: np.ndarray, z0: float = 0.0) -> np.ndarray | None:
    """Пересечение луча с горизонтальной плоскостью z = z0."""
    if abs(direction[2]) < 1e-6:
        return None
    s = (z0 - origin[2]) / direction[2]
    if s <= 0:
        return None
    return origin + s * direction


# ---------------------------------------------------------------------------
# Эпиполярная геометрия
# ---------------------------------------------------------------------------

def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def essential_between(a: CameraModel, b: CameraModel) -> np.ndarray:
    """E такая, что x_b^T E x_a = 0 для нормализованных координат."""
    R_rel = b.R @ a.R.T
    t_rel = b.t.reshape(3) - R_rel @ a.t.reshape(3)
    return _skew(t_rel) @ R_rel


def symmetric_epipolar_px(E: np.ndarray, na: np.ndarray, nb: np.ndarray, fa: float, fb: float) -> np.ndarray:
    """Симметричное эпиполярное расстояние (пиксели) для пар нормализованных точек (N,2)."""
    ha = np.hstack([na, np.ones((len(na), 1))])
    hb = np.hstack([nb, np.ones((len(nb), 1))])
    lb = ha @ E.T  # линии в b
    la = hb @ E  # линии в a
    num = np.abs(np.sum(hb * lb, axis=1))
    db = num / np.maximum(np.linalg.norm(lb[:, :2], axis=1), 1e-12) * fb
    da = num / np.maximum(np.linalg.norm(la[:, :2], axis=1), 1e-12) * fa
    return 0.5 * (da + db)


def look_at(center: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """R, t камеры (OpenCV: x вправо, y вниз, z вперёд), стоящей в center и смотрящей на target."""
    center = np.asarray(center, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64) - center
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, dtype=np.float64))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    t = -R @ center
    return R, t
