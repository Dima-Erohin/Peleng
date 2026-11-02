"""Быстрая калибровка камер по доске ChArUco.

Мировая система координат задаётся доской:
  * начало — центр доски (на высоте board.height_m над полом, если доска лежит на столе);
  * X — вдоль длинной стороны доски (по столбцам), Y — поперёк, Z — вверх (доска лежит лицом вверх).

Режимы:
  * extrinsics (положение камеры) — один снимок доски каждой камерой, секунды;
  * intrinsics (оптика) — 10–20 кадров доски под разными углами, делается один раз на телефон.
    Если внутренние параметры не известны, берутся приблизительные по углу обзора, а фокус
    уточняется по тем же кадрам, что и положение (режим «быстро на питче»).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import CameraModel


@dataclass
class BoardDetection:
    corners: np.ndarray  # (N,2) пиксели
    ids: np.ndarray  # (N,)
    obj_board: np.ndarray  # (N,3) в системе доски OpenCV


class Board:
    def __init__(self, cfg: dict):
        # По умолчанию: 5×4 клетки по 50 мм — помещается на A4. На A3 — клетка 70 мм (дальше видно).
        self.squares_x = int(cfg.get("squares_x", 5))
        self.squares_y = int(cfg.get("squares_y", 4))
        self.square_m = float(cfg.get("square_m", 0.05))
        self.marker_m = float(cfg.get("marker_m", 0.037))
        self.height_m = float(cfg.get("height_m", 0.0))
        self.dict_name = cfg.get("dictionary", "DICT_4X4_50")
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dict_name))
        self.board = cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_m, self.marker_m, dictionary)
        # параметры подобраны для мелких (далёких) маркеров
        dp = cv2.aruco.DetectorParameters()
        dp.minMarkerPerimeterRate = 0.01
        dp.adaptiveThreshWinSizeMin = 3
        dp.adaptiveThreshWinSizeStep = 4
        dp.perspectiveRemovePixelPerCell = 8
        dp.polygonalApproxAccuracyRate = 0.05
        cp = cv2.aruco.CharucoParameters()
        cp.minMarkers = 1
        self.detector = cv2.aruco.CharucoDetector(self.board, cp, dp)
        self.all_corners = np.asarray(self.board.getChessboardCorners(), dtype=np.float64).reshape(-1, 3)
        # центр доски в системе доски OpenCV
        self.center_b = np.array([self.squares_x * self.square_m / 2, self.squares_y * self.square_m / 2, 0.0])
        # доска -> мир: X = x_b, Y = -y_b, Z = -z_b (доска лицом вверх), начало в центре, + высота
        self.R_wb = np.diag([1.0, -1.0, -1.0])
        self.t_wb = -self.R_wb @ self.center_b + np.array([0, 0, self.height_m])

    @property
    def size_m(self) -> tuple[float, float]:
        return self.squares_x * self.square_m, self.squares_y * self.square_m

    def to_dict(self) -> dict:
        return {
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_m": self.square_m,
            "marker_m": self.marker_m,
            "height_m": self.height_m,
            "dictionary": self.dict_name,
            "size_m": list(self.size_m),
        }

    def image(self, px_per_square: int = 200, margin_px: int = 40) -> np.ndarray:
        w = self.squares_x * px_per_square + 2 * margin_px
        h = self.squares_y * px_per_square + 2 * margin_px
        return self.board.generateImage((w, h), marginSize=margin_px, borderBits=1)

    def detect(self, image: np.ndarray, min_corners: int = 6) -> BoardDetection | None:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ch_corners, ch_ids, _, _ = self.detector.detectBoard(gray)
        if ch_ids is None or len(ch_ids) < min_corners:
            return None
        ids = ch_ids.reshape(-1).astype(int)
        return BoardDetection(ch_corners.reshape(-1, 2).astype(np.float64), ids, self.all_corners[ids])

    def board_to_world(self, X_b: np.ndarray) -> np.ndarray:
        return np.asarray(X_b).reshape(-1, 3) @ self.R_wb.T + self.t_wb

    def world_corners(self) -> np.ndarray:
        """4 угла доски в мировых координатах (для отрисовки)."""
        w, h = self.size_m
        pts = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], dtype=np.float64)
        return self.board_to_world(pts)


def _not_collinear(det: BoardDetection) -> bool:
    p = det.obj_board[:, :2]
    if len(p) < 4:
        return False
    c = p - p.mean(axis=0)
    s = np.linalg.svd(c, compute_uv=False)
    return s[-1] > 1e-3


# ---------------------------------------------------------------------------
# Внутренние параметры
# ---------------------------------------------------------------------------

def calibrate_intrinsics(dets: list[BoardDetection], size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, float]:
    """Полная калибровка оптики по нескольким видам доски. Возвращает (K, dist, rms)."""
    dets = [d for d in dets if len(d.ids) >= 8 and _not_collinear(d)]
    if len(dets) < 6:
        raise ValueError(f"мало хороших кадров доски: {len(dets)} (нужно ≥ 6)")
    obj = [d.obj_board.astype(np.float32) for d in dets]
    img = [d.corners.astype(np.float32) for d in dets]
    # у телефонов дисторсия основной камеры маленькая: оставляем только k1 — устойчивее на малом числе кадров
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, tuple(size), None, None, flags=flags)
    return K, dist.reshape(-1), float(rms)


def refine_focal(dets: list[BoardDetection], cam: CameraModel, max_change: float = 0.35) -> tuple[np.ndarray, float] | None:
    """Уточнить только фокусное расстояние по 1+ видам доски (главная точка в центре, без дисторсии).

    Работает, если доска видна под наклоном (лежит на полу, камера смотрит сверху-сбоку).
    Возвращает (K, rms) или None, если результат ненадёжен.
    """
    dets = [d for d in dets if len(d.ids) >= 10 and _not_collinear(d)]
    if not dets:
        return None
    obj = [d.obj_board.astype(np.float32) for d in dets]
    img = [d.corners.astype(np.float32) for d in dets]
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
    )
    try:
        rms, K, _, _, _ = cv2.calibrateCamera(obj, img, tuple(cam.size), cam.K.copy(), np.zeros(5), flags=flags)
    except cv2.error:
        return None
    ratio = K[0, 0] / cam.K[0, 0]
    if not (1 - max_change <= ratio <= 1 + max_change) or not math.isfinite(rms):
        return None
    return K, float(rms)


# ---------------------------------------------------------------------------
# Внешние параметры
# ---------------------------------------------------------------------------

@dataclass
class ExtrinsicsResult:
    ok: bool
    R: np.ndarray | None = None
    t: np.ndarray | None = None
    reproj_px: float = float("nan")
    n_points: int = 0
    message: str = ""
    K: np.ndarray | None = None  # если фокус был уточнён


def solve_extrinsics(board: Board, cam: CameraModel, dets: list[BoardDetection], refine_f: bool = False) -> ExtrinsicsResult:
    """Положение камеры относительно доски (а значит — в мире) по одному или нескольким кадрам."""
    dets = [d for d in dets if d is not None and len(d.ids) >= 6 and _not_collinear(d)]
    if not dets:
        return ExtrinsicsResult(False, message="доска не найдена (нужно ≥ 6 углов)")
    obj = np.vstack([d.obj_board for d in dets]).astype(np.float64)
    img = np.vstack([d.corners for d in dets]).astype(np.float64)

    def pnp(K):
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, cam.dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, cam.dist, rvec, tvec)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, cam.dist)
        return rvec, tvec, float(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1).mean())

    base = pnp(cam.K)
    if base is None:
        return ExtrinsicsResult(False, message="solvePnP не сошёлся")
    rvec, tvec, err = base
    new_K = None
    if refine_f:
        # Внимание: по одному виду маленькой доски фокус определяется грубо (ошибка 1–5 %),
        # а ошибка репроекции почти не меняется — проверить качество по ней нельзя.
        # Поэтому режим выключен по умолчанию; лучше один раз откалибровать оптику (calibrate_intrinsics).
        r = refine_focal(dets, cam)
        if r is not None:
            alt = pnp(r[0])
            if alt is not None:
                rvec, tvec, err = alt
                new_K = r[0]
    R_cb, _ = cv2.Rodrigues(rvec)
    t_cb = tvec.reshape(3)
    # X_c = R_cb X_b + t_cb,  X_b = R_wb^T (X_w - t_wb)
    R_cw = R_cb @ board.R_wb.T
    t_cw = t_cb - R_cw @ board.t_wb
    return ExtrinsicsResult(True, R_cw, t_cw, err, len(obj), "ok", new_K)
