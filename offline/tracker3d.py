"""Причинный 3D-трекинг одного объекта по детекциям в нескольких камерах с известной позой.

Состояние фильтра: x = [p (3), v (3)] (модель постоянной скорости, рабочая система, метры) или
[p, v, a] (постоянное ускорение, model="ca"). Шум процесса — белый шум рывка/ускорения с σ_a.

На каждом кадре k (используются только детекции кадров <= k):
  1. Прогноз на dt = 1/fps.
  2. Гейтинг: для каждой детекции каждой камеры — расстояние Махаланобиса между пикселем детекции
     и прогнозом (S = J P J^T + R_px). Ложные детекции далеко от прогноза отбрасываются.
  3. Ассоциация между камерами: гипотезы строятся из пар камер (эпиполярное расстояние ->
     двухвидовая триангуляция -> проверка ошибки репроекции), затем жадно добавляются согласованные
     детекции остальных камер. Лучшая гипотеза: больше камер, ближе к прогнозу, выше score.
  4. Обновление:
       * >= 2 камер («tri»)  — итерационный EKF по пиксельным измерениям всех камер гипотезы
         (то же самое, что триангуляция, но с учётом прогноза и честной ковариацией);
       * 1 камера («mono»)    — итерационный EKF по одному пикселю: измерение уточняет положение поперёк
         луча, а глубину держит прогноз (модель движения). Дополнительно (size_range=True) — измерение
         дальности по размеру рамки: физический размер объекта для каждой камеры запоминается, пока
         объект виден в >= 2 камерах (дальность известна), и затем даёт глубину d ≈ f·H/h_px;
       * 0 камер («predict»)   — только прогноз.
  5. Инициализация / перезахват: если трека нет или он «потерян» (σ положения велика, давно нет
     обновлений), либо устойчивая многокамерная гипотеза N кадров подряд противоречит треку —
     трек создаётся заново из лучшей многокамерной гипотезы.

Выход кадра k — апостериорная оценка на кадре k (никаких проходов назад).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .detect_base import Detection

CHI2_2 = 11.83  # 99.73 % для 2 степеней свободы
CHI2_3 = 14.16  # 99.73 % для 3 степеней свободы


@dataclass
class TrackerConfig:
    fps: float = 30.0
    model: str = "cv"  # "cv" — постоянная скорость, "ca" — постоянное ускорение
    accel_std: float = 3.0  # σ ускорения (cv) или рывка (ca), м/с² (м/с³)
    point: str = "center"  # какую точку рамки измерять: center | bottom | top | detector
    px_sigma0: float = 2.0  # σ пикселя = px_sigma0 + px_sigma_rel * размер рамки
    px_sigma_rel: float = 0.08
    px_sigma_rel_motion: float = 0.15  # для пятен вычитания фона (рамка «гуляет» сильнее)
    min_score: float = 0.05
    gate_chi2: float = CHI2_2 * 2.0  # гейт пикселя (Махаланобис²)
    gate3_chi2: float = CHI2_3 * 2.0
    pair_reproj_rel: float = 0.35  # допустимая ошибка репроекции при ассоциации: 8 px + доля размера рамки
    pair_reproj_px: float = 8.0
    init_min_score: float = 0.3  # средний score гипотезы для создания трека
    init_min_views: int = 2
    init_vel_std: float = 15.0
    lost_pos_std: float = 60.0  # σ положения (м), после которой трек считается потерянным
    lost_after_s: float = 8.0  # или столько секунд без обновлений
    reinit_frames: int = 8  # столько кадров подряд сильная гипотеза вне гейта -> перезахват
    reinit_min_views: int = 2
    size_range: bool = True  # измерение дальности по размеру рамки в режиме одной камеры
    size_range_rel_std: float = 0.25
    size_learn_alpha: float = 0.1
    iekf_iters: int = 5
    max_speed: float = 80.0  # м/с — ограничение скорости гипотез при инициализации (защита от мусора)
    # модель измерения: "point" — точка рамки (центр), "box" — все 4 стороны рамки как проекция 3D-параллелепипеда
    # (центр p, курс по скорости, размеры L/W/H — в состоянии фильтра, оцениваются по ходу). Модель «box» даёт
    # дальность и по одной камере (через размеры), и честную точку «перёд объекта» (anchor).
    meas: str = "point"
    box_prior: tuple = (6.0, 2.5, 3.0)  # априорные длина/ширина/высота (м)
    box_prior_std: float = 4.0
    box_dims_walk: float = 0.003  # σ случайного блуждания размеров за кадр, м
    box_side_sigma0: float = 2.0  # σ стороны рамки = sigma0 + rel * размер рамки
    box_side_sigma_rel: float = 0.05
    box_side_gate: float = 16.0  # стороны с (r²/S) больше — отбрасываются (заслонение, обрезка)
    box_border_px: float = 3.0
    box_learn_min_speed: float = 2.0  # м/с  # стороны рамки у края кадра не используются (объект обрезан)
    anchor: str = "center"  # выходная точка: center | front | front_bottom | front_top | bottom | top (box: по размерам)
    out_offset_fwd_m: float = 0.0  # смещение выходной точки вдоль направления движения (по горизонтали)
    out_offset_up_m: float = 0.0  # и вверх (например, перёд/верх кабины вместо центра рамки)


@dataclass
class FrameResult:
    frame: int
    pos: np.ndarray | None  # выходная точка (рабочая система, м) или None до инициализации
    vel: np.ndarray | None
    mode: str  # init | tri | mono | predict | none
    n_cams: int
    cams: list[int]
    sigma_m: float  # σ положения (корень из среднего собственного значения ковариации), м
    reproj_px: float = float("nan")
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Проекция и якобиан (CameraModel без дисторсии: K, R, t)
# ---------------------------------------------------------------------------

def project_jac(cam, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Пиксель (2,), якобиан d(u,v)/dp (2,3), глубина."""
    xc = cam.R @ p + cam.t.reshape(3)
    z = xc[2]
    fx, fy, cx, cy = cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]
    zs = z if abs(z) > 1e-6 else 1e-6
    uv = np.array([fx * xc[0] / zs + cx, fy * xc[1] / zs + cy])
    Jc = np.array([[fx / zs, 0, -fx * xc[0] / zs ** 2], [0, fy / zs, -fy * xc[1] / zs ** 2]])
    return uv, Jc @ cam.R, z


def det_point(d: Detection, mode: str) -> np.ndarray:
    x1, y1, x2, y2 = d.bbox
    if mode == "bottom":
        return np.array([(x1 + x2) / 2, y2])
    if mode == "top":
        return np.array([(x1 + x2) / 2, y1])
    if mode == "detector":
        return np.asarray(d.point_px, dtype=np.float64)
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def triangulate_ls(cams: list, pts: list[np.ndarray], sig: list[float], X0: np.ndarray | None = None, iters: int = 8):
    """Триангуляция: DLT -> уточнение Гаусса–Ньютона по взвешенной пиксельной ошибке.

    Возвращает (X, cov(3x3), ошибки репроекции px) или (None, None, None)."""
    if X0 is None:
        rows = []
        for c, uv, s in zip(cams, pts, sig):
            n = np.linalg.solve(c.K, np.array([uv[0], uv[1], 1.0]))
            P = np.hstack([c.R, c.t.reshape(3, 1)])
            rows.append((n[0] * P[2] - P[0]) / s)
            rows.append((n[1] * P[2] - P[1]) / s)
        _, _, Vt = np.linalg.svd(np.asarray(rows))
        Xh = Vt[-1]
        if abs(Xh[3]) < 1e-12:
            return None, None, None
        X = Xh[:3] / Xh[3]
    else:
        X = np.array(X0, dtype=np.float64)
    for _ in range(iters):
        JtJ = np.zeros((3, 3))
        Jtr = np.zeros(3)
        for c, uv, s in zip(cams, pts, sig):
            h, J, z = project_jac(c, X)
            if z <= 0:
                return None, None, None
            r = (uv - h) / s
            Jw = J / s
            JtJ += Jw.T @ Jw
            Jtr += Jw.T @ r
        try:
            dx = np.linalg.solve(JtJ + 1e-9 * np.eye(3), Jtr)
        except np.linalg.LinAlgError:
            return None, None, None
        X = X + dx
        if np.linalg.norm(dx) < 1e-4:
            break
    errs = []
    for c in cams:
        h, _, z = project_jac(c, X)
        if z <= 0:
            return None, None, None
    for c, uv in zip(cams, pts):
        errs.append(float(np.linalg.norm(project_jac(c, X)[0] - uv)))
    try:
        cov = np.linalg.inv(JtJ)
    except np.linalg.LinAlgError:
        cov = np.eye(3) * 1e6
    return X, cov, np.array(errs)


# ---------------------------------------------------------------------------
# Трекер
# ---------------------------------------------------------------------------

class Tracker3D:
    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.nk = 9 if self.cfg.model == "ca" else 6  # кинематическая часть состояния
        self.box = self.cfg.meas == "box"
        self.nx = self.nk + (3 if self.box else 0)  # + размеры L, W, H
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.last_update_frame = -10 ** 9
        self.frame = -1
        self.reinit_count = 0
        self.size_phys: dict[int, float] = {}  # cam_id -> физическая высота объекта (м) по рамке этой камеры
        self.history: list[FrameResult] = []
        self.heading = np.zeros(3)

    # --- модель движения ---
    def _FQ(self, dt: float):
        F, Q = self._FQ_kin(dt)
        if not self.box:
            return F, Q
        n = self.nx
        F2, Q2 = np.eye(n), np.zeros((n, n))
        F2[: self.nk, : self.nk] = F
        Q2[: self.nk, : self.nk] = Q
        Q2[self.nk:, self.nk:] = np.eye(3) * self.cfg.box_dims_walk ** 2
        return F2, Q2

    def _FQ_kin(self, dt: float):
        q = self.cfg.accel_std ** 2
        if self.nk == 6:
            F = np.eye(6)
            F[:3, 3:] = np.eye(3) * dt
            G = np.vstack([np.eye(3) * 0.5 * dt ** 2, np.eye(3) * dt])
        else:
            F = np.eye(9)
            F[:3, 3:6] = np.eye(3) * dt
            F[:3, 6:] = np.eye(3) * 0.5 * dt ** 2
            F[3:6, 6:] = np.eye(3) * dt
            G = np.vstack([np.eye(3) * dt ** 3 / 6, np.eye(3) * 0.5 * dt ** 2, np.eye(3) * dt])
        return F, G @ G.T * q

    def predict(self, dt: float) -> None:
        F, Q = self._FQ(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def _init(self, X: np.ndarray, cov: np.ndarray, frame: int, dims=None) -> None:
        c = self.cfg
        self.x = np.zeros(self.nx)
        self.x[:3] = X
        P = np.zeros((self.nx, self.nx))
        P[:3, :3] = cov + np.eye(3) * 0.25
        P[3:6, 3:6] = np.eye(3) * c.init_vel_std ** 2
        P[5, 5] = (c.init_vel_std * 0.5) ** 2
        if self.nk == 9:
            P[6:9, 6:9] = np.eye(3) * 4.0
        if self.box:
            self.x[self.nk:] = dims if dims is not None else np.array(c.box_prior, dtype=float)
            P[self.nk:, self.nk:] = np.eye(3) * (c.box_prior_std ** 2 if dims is None else 0.5 ** 2)
        self.P = P
        self.last_update_frame = frame
        self.reinit_count = 0

    @property
    def pos_sigma(self) -> float:
        if self.P is None:
            return float("inf")
        return float(math.sqrt(max(np.trace(self.P[:3, :3]) / 3.0, 0.0)))

    def _lost(self) -> bool:
        if self.x is None:
            return True
        c = self.cfg
        return self.pos_sigma > c.lost_pos_std or (self.frame - self.last_update_frame) > c.lost_after_s * c.fps

    # --- измерения ---
    def _sigma_px(self, d: Detection) -> float:
        c = self.cfg
        rel = c.px_sigma_rel_motion if d.cls == "motion" else c.px_sigma_rel
        return c.px_sigma0 + rel * d.size_px

    # --- модель «рамка = проекция параллелепипеда» ---
    def _heading_of(self, x: np.ndarray) -> np.ndarray:
        vh = np.array([x[3], x[4], 0.0])
        nv = np.linalg.norm(vh)
        if nv > 1.0:
            return vh / nv
        if np.linalg.norm(self.heading) > 0:
            return self.heading
        return np.array([1.0, 0.0, 0.0])

    def _box_corners(self, x: np.ndarray) -> np.ndarray:
        f = self._heading_of(x)
        lft = np.array([-f[1], f[0], 0.0])
        L, Wd, Hd = np.maximum(x[self.nk:self.nk + 3], 0.1)
        sg = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], dtype=float) * 0.5
        return x[:3] + sg[:, :1] * L * f + sg[:, 1:2] * Wd * lft + sg[:, 2:3] * Hd * np.array([0, 0, 1.0])

    def _box_h(self, cam, x: np.ndarray):
        Cn = self._box_corners(x)
        xc = Cn @ cam.R.T + cam.t.reshape(3)
        if (xc[:, 2] <= 0.5).any():
            return None
        u = cam.K[0, 0] * xc[:, 0] / xc[:, 2] + cam.K[0, 2]
        v = cam.K[1, 1] * xc[:, 1] / xc[:, 2] + cam.K[1, 2]
        return np.array([u.min(), v.min(), u.max(), v.max()])

    def _box_h_jac(self, cam, x: np.ndarray):
        h0 = self._box_h(cam, x)
        if h0 is None:
            return None, None
        J = np.zeros((4, self.nx))
        idx = list(range(6))
        # размеры уточняем, только когда курс надёжен (объект едет), иначе ориентация параллелепипеда случайна
        vh = math.hypot(x[3], x[4])
        if vh > self.cfg.box_learn_min_speed and self.P is not None and math.sqrt(self.P[3, 3] + self.P[4, 4]) < 0.5 * vh:
            idx += list(range(self.nk, self.nx))
        for i in idx:
            dx = np.zeros(self.nx)
            e = 1e-3 * max(1.0, abs(x[i])) if i >= 3 else 1e-3
            dx[i] = e
            h1 = self._box_h(cam, x + dx)
            if h1 is None:
                return None, None
            J[:, i] = (h1 - h0) / e
        return h0, J

    def _box_sides(self, cam, d: Detection, s: float) -> list[int]:
        """Какие стороны рамки использовать: не у края кадра и согласованные с прогнозом."""
        c = self.cfg
        W, H = cam.size
        b = d.bbox
        ok = [b[0] > c.box_border_px, b[1] > c.box_border_px, b[2] < W - c.box_border_px, b[3] < H - c.box_border_px]
        hb, Hb = self._box_h_jac(cam, self.x)
        if hb is None:
            return []
        out = []
        for j in range(4):
            if not ok[j]:
                continue
            Sj = float(Hb[j] @ self.P @ Hb[j]) + s * s
            if (b[j] - hb[j]) ** 2 / Sj <= c.box_side_gate:
                out.append(j)
        return out

    def _box_sigma(self, d: Detection) -> float:
        c = self.cfg
        rel = c.box_side_sigma_rel * (2.0 if d.cls == "motion" else 1.0)
        return c.box_side_sigma0 + rel * d.size_px

    def _update(self, cams_dets: list[tuple]) -> None:
        """Обновление по набору (камера, детекция, uv, σ_px): точечная модель или модель параллелепипеда."""
        if not self.box:
            self._iekf([(cam, uv, s) for cam, d, uv, s in cams_dets])
            return
        bm = []
        for cam, d, uv, s in cams_dets:
            sb = self._box_sigma(d)
            sides = self._box_sides(cam, d, sb)
            if sides:
                bm.append((cam, np.asarray(d.bbox, dtype=float), sb, sides))
        if not bm:  # все стороны отброшены — хотя бы центр
            self._iekf([(cam, uv, s) for cam, d, uv, s in cams_dets])
            return
        self._iekf([], box_meas=bm)

    def _iekf(self, meas: list[tuple], extra_lin: list[tuple] | None = None, box_meas: list[tuple] | None = None) -> None:
        """Итерационный EKF. meas: [(cam, uv(2,), sigma_px)]; extra_lin: [(h_row(3,), z, sigma)] — линейные
        измерения положения (дальность по размеру)."""
        x0, P0 = self.x.copy(), self.P.copy()
        xi = x0.copy()
        n = self.nx
        for _ in range(max(1, self.cfg.iekf_iters)):
            Hs, rs, Rs = [], [], []
            for cam, zb, sb, sides in box_meas or []:
                hb, Hb = self._box_h_jac(cam, xi)
                if hb is None:
                    continue
                for j in sides:
                    Hs.append(Hb[j:j + 1])
                    rs.append(np.array([zb[j] - hb[j]]) - Hb[j:j + 1] @ (x0 - xi))
                    Rs.append(sb * sb)
            for cam, uv, s in meas:
                h, J, z = project_jac(cam, xi[:3])
                if z <= 0.1:
                    continue
                H = np.zeros((2, n))
                H[:, :3] = J
                Hs.append(H)
                rs.append(uv - h - H @ (x0 - xi))
                Rs += [s * s, s * s]
            for hrow, zval, s in extra_lin or []:
                H = np.zeros((1, n))
                H[0, :3] = hrow
                Hs.append(H)
                rs.append(np.array([zval - hrow @ xi[:3]]) - H @ (x0 - xi))
                Rs.append(s * s)
            if not Hs:
                return
            H = np.vstack(Hs)
            r = np.concatenate(rs)
            Rm = np.diag(Rs)
            S = H @ P0 @ H.T + Rm
            Kg = np.linalg.solve(S, H @ P0).T
            xn = x0 + Kg @ r
            if np.linalg.norm(xn[:3] - xi[:3]) < 1e-4:
                xi = xn
                break
            xi = xn
        I_KH = np.eye(n) - Kg @ H
        self.x = xi
        self.P = I_KH @ P0 @ I_KH.T + Kg @ Rm @ Kg.T  # форма Джозефа

    # --- гипотезы ---
    def _hypotheses(self, cams: list, cands: list[list[tuple]]):
        """Многокамерные гипотезы. cands[i] = [(det_index, uv, sigma, det)]. Возвращает список
        (X, cov, views=[(i, j)], errs)."""
        c = self.cfg
        hyps = []
        seen = set()
        m = len(cams)
        for a in range(m):
            for b in range(a + 1, m):
                for ia, (_, uva, sa, da) in enumerate(cands[a]):
                    for ib, (_, uvb, sb, db) in enumerate(cands[b]):
                        X, cov, errs = triangulate_ls([cams[a], cams[b]], [uva, uvb], [sa, sb])
                        if X is None:
                            continue
                        tol = [c.pair_reproj_px + c.pair_reproj_rel * d.size_px for d in (da, db)]
                        if errs[0] > tol[0] or errs[1] > tol[1]:
                            continue
                        views = {a: ia, b: ib}
                        # жадно добавить остальные камеры
                        for o in range(m):
                            if o in views or not cands[o]:
                                continue
                            best = None
                            for io, (_, uvo, so, do) in enumerate(cands[o]):
                                h, _, z = project_jac(cams[o], X)
                                if z <= 0:
                                    continue
                                e = float(np.linalg.norm(h - uvo))
                                if e <= c.pair_reproj_px + c.pair_reproj_rel * do.size_px and (best is None or e < best[0]):
                                    best = (e, io)
                            if best is not None:
                                views[o] = best[1]
                        key = tuple(sorted(views.items()))
                        if key in seen:
                            continue
                        seen.add(key)
                        if len(views) > 2:
                            idx = sorted(views)
                            X2, cov2, errs2 = triangulate_ls([cams[i] for i in idx], [cands[i][views[i]][1] for i in idx],
                                                             [cands[i][views[i]][2] for i in idx], X0=X)
                            if X2 is not None:
                                X, cov, errs = X2, cov2, errs2
                        hyps.append((X, cov, sorted(views.items()), errs))
        return hyps

    # --- главный шаг ---
    def step(self, frame: int, cams: list, dets: list[list[Detection]], cam_ids: list[int] | None = None) -> FrameResult:
        """cams[i] — CameraModel камеры i на этом кадре, dets[i] — её детекции."""
        c = self.cfg
        cam_ids = cam_ids or list(range(len(cams)))
        dt = 1.0 / c.fps if self.frame >= 0 else 0.0
        self.frame = frame
        if self.x is not None:
            self.predict(dt)
        all_cands = []
        for cam, dl in zip(cams, dets):
            all_cands.append([(j, det_point(d, c.point), self._sigma_px(d), d) for j, d in enumerate(dl) if d.score >= c.min_score])

        lost = self._lost()
        mode, used, reproj = "predict" if self.x is not None else "none", [], float("nan")
        info: dict = {}

        # гейтинг по прогнозу
        gated = [[] for _ in cams]
        if not lost:
            Pp = self.P[:3, :3]
            for i, (cam, cl) in enumerate(zip(cams, all_cands)):
                if not cl:
                    continue
                h, J, z = project_jac(cam, self.x[:3])
                if z <= 0.1:
                    continue
                for cand in cl:
                    _, uv, s, d = cand
                    S = J @ Pp @ J.T + np.eye(2) * s * s
                    r = uv - h
                    d2 = float(r @ np.linalg.solve(S, r))
                    if d2 <= c.gate_chi2:
                        gated[i].append(cand + (d2,))
                gated[i].sort(key=lambda t: t[4])
                gated[i] = gated[i][:3]

        # сильная гипотеза среди всех детекций (для инициализации и перезахвата)
        strong = None
        hyps_all = self._hypotheses(cams, [cl[:4] for cl in all_cands]) if any(all_cands) else []
        good = []
        for X, cov, views, errs in hyps_all:
            sc = float(np.mean([all_cands[i][j][3].score for i, j in views]))
            if len(views) >= c.init_min_views and sc >= c.init_min_score:
                good.append((len(views), sc, X, cov, views, errs))
        if good:
            good.sort(key=lambda g: (-g[0], -g[1]))
            strong = good[0]

        if lost:
            if strong is not None:
                n, sc, X, cov, views, errs = strong
                self._init(X, cov, frame, self.x[self.nk:].copy() if (self.box and self.x is not None) else None)
                mode, used, reproj = "init", [cam_ids[i] for i, _ in views], float(np.mean(errs))
                self._learn_size(cams, all_cands, views, cam_ids)
        else:
            hyps = self._hypotheses(cams, [[g[:4] for g in gl] for gl in gated])
            best = None
            Pp = self.P[:3, :3]
            for X, cov, views, errs in hyps:
                r = X - self.x[:3]
                d3 = float(r @ np.linalg.solve(Pp + cov, r))
                if d3 > c.gate3_chi2:
                    continue
                sc = float(np.mean([gated[i][j][3].score for i, j in views]))
                key = (len(views), -d3 + 5 * sc)
                if best is None or key > best[0]:
                    best = (key, X, cov, views, errs)
            if best is not None:
                _, X, cov, views, errs = best
                self._update([(cams[i], gated[i][j][3], gated[i][j][1], gated[i][j][2]) for i, j in views])
                mode, used, reproj = "tri", [cam_ids[i] for i, _ in views], float(np.mean(errs))
                self.last_update_frame = frame
                self._learn_size(cams, gated, views, cam_ids)
            else:
                # одна камера: лучшая по Махаланобису детекция (с учётом score)
                bestm = None
                for i, gl in enumerate(gated):
                    if gl:
                        g = gl[0]
                        key = g[4] - 2.0 * g[3].score
                        if bestm is None or key < bestm[0]:
                            bestm = (key, i, g)
                if bestm is not None:
                    _, i, g = bestm
                    if self.box:
                        extra = None
                        self._update([(cams[i], g[3], g[1], g[2])])
                    else:
                        extra = self._size_measure(cams[i], cam_ids[i], g[3]) if c.size_range else None
                        self._iekf([(cams[i], g[1], g[2])], extra)
                    mode, used = "mono", [cam_ids[i]]
                    reproj = float(np.linalg.norm(project_jac(cams[i], self.x[:3])[0] - g[1]))
                    self.last_update_frame = frame
                    if extra:
                        info["size_range"] = True
            # перезахват: сильная гипотеза далеко от трека несколько кадров подряд
            if strong is not None and strong[0] >= c.reinit_min_views:
                X = strong[2]
                r = X - self.x[:3]
                d3 = float(r @ np.linalg.solve(self.P[:3, :3] + strong[3], r))
                if d3 > c.gate3_chi2 and (mode != "tri" or len(used) < strong[0]):
                    self.reinit_count += 1
                else:
                    self.reinit_count = 0
                if self.reinit_count >= c.reinit_frames:
                    n, sc, X, cov, views, errs = strong
                    self._init(X, cov, frame, self.x[self.nk:].copy() if (self.box and self.x is not None) else None)
                    mode, used, reproj = "init", [cam_ids[i] for i, _ in views], float(np.mean(errs))
                    info["reinit"] = True
            else:
                self.reinit_count = 0

        res = self._result(frame, mode, used, reproj, info)
        self.history.append(res)
        return res

    def _learn_size(self, cams, cands, views, cam_ids) -> None:
        """Запомнить физическую высоту объекта по рамке каждой камеры (когда дальность известна по >= 2 камерам)."""
        if self.x is None:
            return
        a = self.cfg.size_learn_alpha
        for i, j in views:
            d = cands[i][j][3]
            h_px = max(1.0, d.bbox[3] - d.bbox[1])
            z = float(cams[i].R[2] @ self.x[:3] + cams[i].t.reshape(3)[2])
            if z <= 0:
                continue
            Hm = h_px * z / cams[i].K[1, 1]
            cid = cam_ids[i]
            self.size_phys[cid] = Hm if cid not in self.size_phys else (1 - a) * self.size_phys[cid] + a * Hm

    def _size_measure(self, cam, cid: int, d: Detection):
        """Линейное измерение глубины (ось z камеры) по высоте рамки: z = f·H/h_px."""
        Hm = self.size_phys.get(cid)
        if Hm is None:
            return None
        h_px = d.bbox[3] - d.bbox[1]
        if h_px < 4:
            return None
        zm = cam.K[1, 1] * Hm / h_px
        s = self.cfg.size_range_rel_std * zm
        # гейт: не принимать дальность, сильно противоречащую прогнозу
        zp = float(cam.R[2] @ self.x[:3] + cam.t.reshape(3)[2])
        varp = float(cam.R[2] @ self.P[:3, :3] @ cam.R[2])
        if (zm - zp) ** 2 / (varp + s * s) > CHI2_2:
            return None
        return [(cam.R[2].copy(), zm - float(cam.t.reshape(3)[2]), s)]

    def _result(self, frame, mode, used, reproj, info) -> FrameResult:
        if self.x is None:
            return FrameResult(frame, None, None, mode, 0, [], float("inf"), reproj, info)
        p = self.x[:3].copy()
        v = self.x[3:6].copy()
        c = self.cfg
        vh = np.array([v[0], v[1], 0.0])
        nv = np.linalg.norm(vh)
        if nv > 1.0:  # курс по горизонтальной скорости; при остановке держим последний
            self.heading = vh / nv
        if self.box and c.anchor != "center":
            L, Wd, Hd = np.maximum(self.x[self.nk:self.nk + 3], 0.0)
            if c.anchor.startswith("front"):
                p = p + self.heading * L / 2
            if c.anchor.endswith("bottom"):
                p[2] -= Hd / 2
            elif c.anchor.endswith("top"):
                p[2] += Hd / 2
        if c.out_offset_fwd_m or c.out_offset_up_m:
            p = p + self.heading * c.out_offset_fwd_m
            p[2] += c.out_offset_up_m
        return FrameResult(frame, p, v, mode, len(used), used, self.pos_sigma, reproj, info)
