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
    accel_std: float = 3.0  # σ ускорения (cv) или рывка (ca), м/с² (м/с³), по горизонтали
    accel_std_z: float | None = None  # то же по вертикали (None — как по горизонтали; для наземной техники — меньше)
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
    object_size_m: float | None = None  # априорный размер объекта (м) по size_dim: дальность по размеру без обучения
    size_dim: str = "height"  # какой размер рамки сравнивать: height | width | geom
    object_size_rel_std: float = 0.5  # относительная σ дальности по априорному размеру
    mono_init: bool = True  # создавать трек по одной камере (по выученному для камеры или априорному размеру)
    prior_init_first_only: bool = False  # априорный размер — только для самого первого создания трека
    reacq_prior: bool = True  # перезахват по одной камере с априорным размером (иначе — только с выученным)
    size_learn_alpha: float = 0.1
    size_rel_min: float = 0.04  # σ дальности по выученному размеру сразу после обучения (доля дальности)
    size_rel_growth: float | None = None  # рост σ в секунду с последнего обучения (None — постоянная size_range_rel_std)
    iekf_iters: int = 5
    # прогноз по дуге (поворот скорости с оценённой угловой скоростью) — для участков без камер на поворотах
    turn_predict: bool = False
    turn_alpha: float = 0.05
    turn_tau_s: float = 3.0
    turn_max: float = 0.5  # рад/с
    turn_min_speed: float = 3.0
    max_speed: float = 60.0  # м/с — физическое ограничение скорости (защита от расхождения фильтра)
    # модель измерения: "point" — точка рамки (центр), "box" — все 4 стороны рамки как проекция 3D-параллелепипеда
    # (центр p, курс по скорости, размеры L/W/H — в состоянии фильтра, оцениваются по ходу). Модель «box» даёт
    # дальность и по одной камере (через размеры), и честную точку «перёд объекта» (anchor).
    meas: str = "point"
    box_prior: tuple = (6.0, 2.5, 3.0)  # априорные длина/ширина/высота (м)
    box_prior_std: float = 4.0
    dims_prior_std: tuple = (10.0, 10.0, 10.0)
    # откуда брать смещение «перёд» (anchor=front): "dims" — L/2 по оценке размеров (DimsEstimator);
    # "detector_point" — медиана (точка детектора, триангулированная по >=2 камерам) − центр, вдоль курса и по высоте
    front_source: str = "dims"
    # оценка размеров: "lsq" — робастный МНК L/W/H по всем сторонам рамок; "median_L" — медиана оценок длины
    # по ширине рамки в видах «сбоку» (W/H — априорные box_prior)
    dims_mode: str = "lsq"
    size_from_dims: bool = False  # дальность по одной камере — по ожидаемой рамке параллелепипеда (учитывает ракурс)
    front_min_samples: int = 10  # σ априорных L/W/H для DimsEstimator (anchor=front в режиме point)
    box_dims_walk: float = 0.003  # σ случайного блуждания размеров за кадр, м
    box_side_sigma0: float = 2.0  # σ стороны рамки = sigma0 + rel * размер рамки
    box_side_sigma_rel: float = 0.05
    box_side_gate: float = 16.0  # стороны с (r²/S) больше — отбрасываются (заслонение, обрезка)
    box_border_px: float = 3.0
    box_learn_min_speed: float = 2.0  # м/с
    box_dims_min: float = 0.3
    box_dims_max: float = 30.0
    box_point_classes: tuple = ("motion",)
    weak_weight: float = 0.35
    # Слияние детекторов: детекции из primary_sources (например, "truck_pose") измеряют точку состояния
    # напрямую (точка cfg.point, обычно "kp:pivot"); остальные («запасные») — точку secondary_point, сдвинутую
    # относительно состояния на secondary_offset (вперёд по курсу, влево, вверх; м). Пусто — все детекции одинаковы.
    primary_sources: tuple = ()
    secondary_point: str = "center"
    secondary_offset: tuple = (0.0, 0.0, 0.0)
    px_sigma_rel_primary: float = 0.03
    init_confirm_frames: int = 1  # сильная гипотеза должна держаться столько кадров подряд, чтобы создать трек
    init_confirm_dist_m: float = 15.0  # вес «слабых» детекций (motion) при выборе гипотезы  # классы, для которых в модели box используется только центр рамки  # стороны рамки у края кадра не используются (объект обрезан)
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
    if mode.startswith("kp:"):
        kp = d.keypoint(mode[3:])
        if kp is not None:
            return np.array(kp)
        return np.asarray(d.point_px, dtype=np.float64)  # нет ключевой точки — точка детектора
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

class DimsEstimator:
    """Причинная оценка размеров объекта (длина L, ширина W, высота H) по рамкам — для выходной точки «перёд».

    Объект — параллелепипед, ориентированный по курсу (горизонтальная скорость фильтра). Пока объект виден в >= 2
    камерах и едет (курс надёжен), в буфер складываются наблюдения (камера, рамка, годные стороны, центр p и курс f
    из фильтра на ЭТОМ кадре). Раз в несколько кадров размеры уточняются робастным МНК (Гаусс–Ньютон + Хьюбер)
    по сторонам рамок: сторона = min/max проекций 8 вершин. Используются только прошлые кадры — причинно.
    Стороны рамок у края кадра (объект обрезан) не используются; пятна вычитания фона — тоже (тень, «след»)."""

    def __init__(self, prior=(6.0, 2.5, 3.0), prior_std=10.0, max_obs=200, every=10, border=3.0, side_sigma=(2.0, 0.05),
                 mode: str = "lsq"):
        self.dims = np.array(prior, dtype=float)
        self.prior = np.array(prior, dtype=float)
        self.prior_std = np.broadcast_to(np.asarray(prior_std, dtype=float), (3,)).copy()
        self.obs: list = []
        self.max_obs = max_obs
        self.every = every
        self.border = border
        self.side_sigma = side_sigma
        self._since = 0
        self.n_solves = 0
        self.mode = mode
        self.min_side_sin = 0.5
        self.L_samples: list = []

    @staticmethod
    def box_bbox(cam, p, f, dims):
        lft = np.array([-f[1], f[0], 0.0])
        L, W, H = dims
        sg = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], dtype=float) * 0.5
        C = p + sg[:, :1] * L * f + sg[:, 1:2] * W * lft + sg[:, 2:3] * H * np.array([0, 0, 1.0])
        xc = C @ cam.R.T + cam.t.reshape(3)
        if (xc[:, 2] <= 0.5).any():
            return None
        u = cam.K[0, 0] * xc[:, 0] / xc[:, 2] + cam.K[0, 2]
        v = cam.K[1, 1] * xc[:, 1] / xc[:, 2] + cam.K[1, 2]
        return np.array([u.min(), v.min(), u.max(), v.max()])

    def add(self, cam, d: Detection, p, f) -> None:
        W, H = cam.size
        b = np.asarray(d.bbox, dtype=float)
        ok = np.array([b[0] > self.border, b[1] > self.border, b[2] < W - self.border, b[3] < H - self.border])
        if ok.sum() < 2:
            return
        s = self.side_sigma[0] + self.side_sigma[1] * d.size_px
        if self.mode == "median_L":
            self._add_L_sample(cam, b, ok, np.array(p, float), np.array(f, float))
            return
        self.obs.append((cam, b, ok, np.array(p, float), np.array(f, float), s))
        if len(self.obs) > self.max_obs:
            self.obs.pop(0)
        self._since += 1
        if self._since >= self.every:
            self._since = 0
            self.solve()

    def _add_L_sample(self, cam, b, ok, p, f) -> None:
        """Одна оценка длины L по ширине рамки (стороны x1, x2 не обрезаны), W и H — априорные.
        Берутся только виды «сбоку» (луч камеры заметно поперёк курса), иначе L по ширине не определяется."""
        if not (ok[0] and ok[2]):
            return
        ray = p - cam.center
        ray[2] = 0.0
        nr = np.linalg.norm(ray)
        if nr < 1e-6 or abs(np.cross(ray / nr, f)[2]) < self.min_side_sin:
            return
        w_obs = b[2] - b[0]

        def width(L):
            h = self.box_bbox(cam, p, f, (L, self.prior[1], self.prior[2]))
            return None if h is None else h[2] - h[0]

        lo, hi = 0.5, 40.0
        wl, wh = width(lo), width(hi)
        if wl is None or wh is None or not (wl <= w_obs <= wh):
            return
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            wm = width(mid)
            if wm is None:
                return
            if wm < w_obs:
                lo = mid
            else:
                hi = mid
        self.L_samples.append(0.5 * (lo + hi))
        if len(self.L_samples) > 3000:
            self.L_samples.pop(0)
        if len(self.L_samples) >= 10:
            self.dims[0] = float(np.median(self.L_samples))

    def solve(self, iters: int = 4) -> None:
        if len(self.obs) < 10:
            return
        x = self.dims.copy()
        for _ in range(iters):
            rows, res = [], []
            for cam, b, ok, p, f, s in self.obs:
                h0 = self.box_bbox(cam, p, f, x)
                if h0 is None:
                    continue
                J = np.zeros((4, 3))
                for i in range(3):
                    dx = np.zeros(3)
                    dx[i] = 0.05
                    h1 = self.box_bbox(cam, p, f, x + dx)
                    if h1 is None:
                        break
                    J[:, i] = (h1 - h0) / 0.05
                r = (b - h0) / s
                w = np.where(np.abs(r) <= 2.0, 1.0, 2.0 / np.maximum(np.abs(r), 1e-9))  # Хьюбер
                m = ok
                rows.append(J[m] / s * np.sqrt(w[m])[:, None])
                res.append(r[m] * np.sqrt(w[m]))
            if not rows:
                return
            A = np.vstack(rows + [np.diag(1.0 / self.prior_std)])
            y = np.concatenate(res + [(self.prior - x) / self.prior_std])
            dx = np.linalg.lstsq(A, y, rcond=None)[0]
            x = np.clip(x + dx, 0.3, 40.0)
            if np.abs(dx).max() < 1e-3:
                break
        self.dims = x
        self.n_solves += 1


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
        self._learn_dims = False
        self._reacq = None
        self._cv_cache: dict = {}
        self._heading_ok = False
        self._confirm: list = []
        self.omega = 0.0
        self._prev_v = None
        self.front_samples: list = []
        self.size_learn_frame: dict[int, int] = {}
        self.dims_est = DimsEstimator(self.cfg.box_prior, self.cfg.dims_prior_std, mode=self.cfg.dims_mode) if (not self.box and self.cfg.anchor != "center") else None

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
        Q = G @ G.T * q
        if self.cfg.accel_std_z is not None:
            # масштаб шума вертикальной компоненты (строки/столбцы z, vz[, az])
            r = (self.cfg.accel_std_z / max(self.cfg.accel_std, 1e-9))
            zi = [2, 5] + ([8] if self.nk == 9 else [])
            sc = np.ones(self.nk)
            sc[zi] = r
            Q = Q * np.outer(sc, sc)
        return F, Q

    def predict(self, dt: float) -> None:
        F, Q = self._FQ(dt)
        c = self.cfg
        if c.turn_predict and abs(self.omega) > 1e-4:
            # «прогноз вдоль дуги»: горизонтальная скорость поворачивается с угловой скоростью ω, оценённой по
            # прошлым кадрам (как едет машина по дороге с поворотом); без измерений ω затухает за turn_tau_s
            since = (self.frame - self.last_update_frame) / c.fps
            w = self.omega * (math.exp(-since / c.turn_tau_s) if since > 0.2 else 1.0)
            a = w * dt
            ca, sa = math.cos(a), math.sin(a)
            Rz = np.eye(self.nx)
            Rz[3:5, 3:5] = [[ca, -sa], [sa, ca]]
            F = Rz @ F
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def _update_omega(self) -> None:
        """Угловая скорость курса по последовательным апостериорным скоростям (только при движении, причинно)."""
        v = self.x[3:5]
        sp = float(np.linalg.norm(v))
        if self._prev_v is not None and sp > self.cfg.turn_min_speed and np.linalg.norm(self._prev_v) > self.cfg.turn_min_speed:
            d = math.atan2(self._prev_v[0] * v[1] - self._prev_v[1] * v[0], float(self._prev_v @ v))
            w = d * self.cfg.fps
            w = max(-self.cfg.turn_max, min(self.cfg.turn_max, w))
            self.omega = (1 - self.cfg.turn_alpha) * self.omega + self.cfg.turn_alpha * w
        self._prev_v = v.copy()

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
        since = self.frame - self.last_update_frame
        # σ большая сразу после создания по одной камере — это нормально; «потерян» = давно нет обновлений
        # или σ велика и обновлений нет уже секунду
        return since > c.lost_after_s * c.fps or (self.pos_sigma > c.lost_pos_std and since > c.fps)

    # --- измерения ---
    def _is_primary(self, d: Detection) -> bool:
        return bool(self.cfg.primary_sources) and d.source in self.cfg.primary_sources

    def _point_of(self, d: Detection) -> np.ndarray:
        if self.cfg.primary_sources and not self._is_primary(d):
            return det_point(d, self.cfg.secondary_point)
        return det_point(d, self.cfg.point)

    def _offset_world(self, d: Detection) -> np.ndarray | None:
        """Сдвиг измеряемой точки детекции относительно точки состояния (мир, м) или None."""
        c = self.cfg
        if not c.primary_sources or self._is_primary(d) or not any(c.secondary_offset):
            return None
        f = self.heading if self._heading_reliable() else np.zeros(3)
        lft = np.array([-f[1], f[0], 0.0])
        return f * c.secondary_offset[0] + lft * c.secondary_offset[1] + np.array([0, 0, 1.0]) * c.secondary_offset[2]

    def _heading_reliable(self) -> bool:
        """Курс известен: объект едет и скорость оценена уверенно (иначе — последний надёжный курс при остановке)."""
        if self.x is None or self.P is None:
            return False
        vh = math.hypot(self.x[3], self.x[4])
        if vh > 3.0 and math.sqrt(self.P[3, 3] + self.P[4, 4]) < 0.3 * vh:
            self._heading_ok = True
        elif vh > 3.0:
            self._heading_ok = False
        return self._heading_ok and np.linalg.norm(self.heading) > 0.5

    def _cv(self, i: int, d: Detection, cams: list):
        """Камера для детекции: при сдвиге o измеряется проекция (p + o) = камера с t' = t + R·o."""
        o = self._offset_world(d)
        if o is None:
            return cams[i]
        cam = cams[i]
        key = (id(cam), round(float(o[0]), 4), round(float(o[1]), 4), round(float(o[2]), 4))
        cv = self._cv_cache.get(key)
        if cv is None:
            from peleng.geometry import CameraModel
            cv = CameraModel(cam.K, cam.dist, cam.size, cam.R, cam.t.reshape(3) + cam.R @ o, dict(cam.meta))
            self._cv_cache[key] = cv
        return cv

    def _det_weight(self, d: Detection) -> float:
        """Вес детекции при выборе гипотезы: пятна вычитания фона слабее детекций нейросети."""
        return self.cfg.weak_weight if d.cls in self.cfg.box_point_classes else 1.0

    def _sigma_px(self, d: Detection) -> float:
        c = self.cfg
        if self._is_primary(d):
            return c.px_sigma0 + c.px_sigma_rel_primary * d.size_px
        if c.primary_sources and any(c.secondary_offset[:2]) and not self._heading_reliable():
            # курс ещё неизвестен -> сдвиг запасной точки не применить: её неопределённость ~ размеру рамки
            return c.px_sigma0 + (c.px_sigma_rel + 0.5) * d.size_px
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
        # курс берётся из скорости, но сторонам рамки НЕ разрешаем менять скорость через курс
        # (иначе ошибка размеров «объясняется» поворотом и скорость скачет): производные только по p и размерам
        idx = list(range(3))
        # размеры уточняем, только когда курс надёжен (объект едет), иначе ориентация параллелепипеда случайна
        vh = math.hypot(x[3], x[4])
        if self._learn_dims and vh > self.cfg.box_learn_min_speed and self.P is not None and math.sqrt(self.P[3, 3] + self.P[4, 4]) < 0.5 * vh:
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
        bm, pm = [], []
        for cam, d, uv, s in cams_dets:
            if d.cls in self.cfg.box_point_classes:
                # пятна вычитания фона захватывают тень и «след» — их стороны ненадёжны, берём только центр
                pm.append((cam, uv, s))
                continue
            sb = self._box_sigma(d)
            sides = self._box_sides(cam, d, sb)
            if sides:
                bm.append((cam, np.asarray(d.bbox, dtype=float), sb, sides))
            else:
                pm.append((cam, uv, s))
        self._learn_dims = len(bm) >= 2  # размеры уточняем только при >= 2 камерах (иначе масштаб неоднозначен)
        self._iekf(pm, box_meas=bm)
        self._learn_dims = False
        k = self.nk
        self.x[k:k + 3] = np.clip(self.x[k:k + 3], self.cfg.box_dims_min, self.cfg.box_dims_max)

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
        sp = float(np.linalg.norm(self.x[3:6]))
        if sp > self.cfg.max_speed:  # физическое ограничение скорости
            self.x[3:6] *= self.cfg.max_speed / sp
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
                        X, cov, errs = triangulate_ls([self._cv(a, da, cams), self._cv(b, db, cams)], [uva, uvb], [sa, sb])
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
                                h, _, z = project_jac(self._cv(o, do, cams), X)
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
                            X2, cov2, errs2 = triangulate_ls([self._cv(i, cands[i][views[i]][3], cams) for i in idx],
                                                             [cands[i][views[i]][1] for i in idx],
                                                             [cands[i][views[i]][2] for i in idx], X0=X)
                            if X2 is not None:
                                X, cov, errs = X2, cov2, errs2
                        hyps.append((X, cov, sorted(views.items()), errs))
        return hyps

    # --- главный шаг ---
    def predict_regions(self, cams: list, n_sigma: float = 3.0) -> list:
        """Ожидаемые рамки объекта на СЛЕДУЮЩЕМ кадре (по прогнозу, состояние не меняется) — для поиска
        детектором в окне (ROI). Возвращает для каждой камеры (x1, y1, x2, y2) или None. Причинно: использует
        только состояние после кадров < k."""
        if self.x is None or self._lost():
            return [None] * len(cams)
        F, Q = self._FQ(1.0 / self.cfg.fps)
        x = F @ self.x
        P = F @ self.P @ F.T + Q
        out = []
        for cam in cams:
            W, H = cam.size
            uv, J, z = project_jac(cam, x[:3])
            if z <= 0.5:
                out.append(None)
                continue
            S = J @ P[:3, :3] @ J.T
            m = n_sigma * np.sqrt(np.maximum(np.diag(S), 0))
            if self.box:
                b = self._box_h(cam, x)
                if b is None:
                    out.append(None)
                    continue
            else:
                s = cam.K[0, 0] * 3.0 / z  # ~3 м объект, если размер неизвестен
                b = np.array([uv[0] - s, uv[1] - s, uv[0] + s, uv[1] + s])
            b = np.array([b[0] - m[0], b[1] - m[1], b[2] + m[0], b[3] + m[1]])
            if b[2] < 0 or b[3] < 0 or b[0] > W or b[1] > H:
                out.append(None)
                continue
            out.append(tuple(float(v) for v in b))
        return out

    def step(self, frame: int, cams: list, dets: list[list[Detection]], cam_ids: list[int] | None = None) -> FrameResult:
        """cams[i] — CameraModel камеры i на этом кадре, dets[i] — её детекции."""
        c = self.cfg
        cam_ids = cam_ids or list(range(len(cams)))
        dt = 1.0 / c.fps if self.frame >= 0 else 0.0
        self.frame = frame
        if self.x is not None:
            self.predict(dt)
        self._cv_cache = {}
        all_cands = []
        for cam, dl in zip(cams, dets):
            all_cands.append([(j, self._point_of(d), self._sigma_px(d), d) for j, d in enumerate(dl) if d.score >= c.min_score])

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
                for cand in cl:
                    _, uv, s, d = cand
                    h, J, z = project_jac(self._cv(i, d, cams), self.x[:3])
                    if z <= 0.1:
                        continue
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

        if lost and strong is not None and c.init_confirm_frames > 1:
            # защита от ложного старта: гипотеза должна держаться несколько кадров подряд примерно на месте
            if self._confirm and self._confirm[-1][0] == frame - 1 and \
                    np.linalg.norm(strong[2] - self._confirm[-1][1]) < c.init_confirm_dist_m:
                self._confirm.append((frame, strong[2]))
            else:
                self._confirm = [(frame, strong[2])]
            if len(self._confirm) < c.init_confirm_frames:
                strong_for_init = None
            else:
                strong_for_init = strong
        else:
            strong_for_init = strong
        if lost:
            if strong_for_init is not None:
                n, sc, X, cov, views, errs = strong_for_init
                self._init(X, cov, frame, self.x[self.nk:].copy() if (self.box and self.x is not None) else None)
                mode, used, reproj = "init", [cam_ids[i] for i, _ in views], float(np.mean(errs))
                info["used_det"] = [(i, all_cands[i][j][0]) for i, j in views]
                self._learn_size(cams, all_cands, views, cam_ids)
            elif c.mono_init and (self.size_phys or c.object_size_m is not None):
                # по одной камере: размер, выученный для неё, или априорный (если разрешено в этот момент)
                prior_ok = c.object_size_m is not None and (self.x is None or not c.prior_init_first_only)
                mono = self._mono_init_candidate(cams, all_cands, cam_ids, prior_ok)
                if mono is not None:
                    i, j, X, cov = mono
                    self._init(X, cov, frame, self.x[self.nk:].copy() if (self.box and self.x is not None) else None)
                    mode, used = "init", [cam_ids[i]]
                    info["used_det"] = [(i, all_cands[i][j][0])]
                    info["mono_init"] = True
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
                wsum = sum(self._det_weight(gated[i][j][3]) for i, j in views)
                key = (round(wsum, 3), -d3 + 5 * sc)
                if best is None or key > best[0]:
                    best = (key, X, cov, views, errs)
            # одна «сильная» детекция (YOLO) важнее согласованной пары слабых (пятна фона)
            if best is not None:
                wbest_mono = max((self._det_weight(gl[0][3]) for gl in gated if gl), default=0.0)
                if wbest_mono > best[0][0]:
                    best = None
            if best is not None:
                _, X, cov, views, errs = best
                self._update([(self._cv(i, gated[i][j][3], cams), gated[i][j][3], gated[i][j][1], gated[i][j][2])
                              for i, j in views])
                mode, used, reproj = "tri", [cam_ids[i] for i, _ in views], float(np.mean(errs))
                info["used_det"] = [(i, gated[i][j][0]) for i, j in views]
                self.last_update_frame = frame
                self._learn_size(cams, gated, views, cam_ids)
                if self.dims_est is not None and len(views) >= 2:
                    vh = math.hypot(self.x[3], self.x[4])
                    if vh > c.box_learn_min_speed and math.sqrt(self.P[3, 3] + self.P[4, 4]) < 0.3 * vh:
                        f = np.array([self.x[3] / vh, self.x[4] / vh, 0.0])
                        for i, j in views:
                            d = gated[i][j][3]
                            if d.cls not in c.box_point_classes and d.score >= 0.3:
                                self.dims_est.add(cams[i], d, self.x[:3], f)
                        if c.front_source == "detector_point":
                            self._learn_front_offset(cams, [gated[i][j][3] for i, j in views],
                                                     [gated[i][j][2] for i, j in views], [i for i, _ in views], f)
            else:
                # одна камера: лучшая по Махаланобису детекция (с учётом score)
                bestm = None
                for i, gl in enumerate(gated):
                    if gl:
                        g = min(gl, key=lambda g: g[4] - 4.0 * self._det_weight(g[3]))
                        key = g[4] - 4.0 * self._det_weight(g[3]) - 2.0 * g[3].score
                        if bestm is None or key < bestm[0]:
                            bestm = (key, i, g)
                if bestm is not None:
                    _, i, g = bestm
                    cv = self._cv(i, g[3], cams)
                    if self.box:
                        extra = None
                        self._update([(cv, g[3], g[1], g[2])])
                    else:
                        extra = self._size_measure(cv, cam_ids[i], g[3]) if c.size_range else None
                        self._iekf([(cv, g[1], g[2])], extra)
                    mode, used = "mono", [cam_ids[i]]
                    info["used_det"] = [(i, g[0])]
                    if extra and cam_ids[i] not in self.size_phys:
                        # глубина держится только на априорном размере объекта — это допущение, а не измерение
                        mode = "mono_size_prior"
                    reproj = float(np.linalg.norm(project_jac(cv, self.x[:3])[0] - g[1]))
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
                    info["used_det"] = [(i, all_cands[i][j][0]) for i, j in views]
                    info["reinit"] = True
            else:
                self.reinit_count = 0
            # перезахват по одной камере: трек только прогнозируется, а устойчивая (несколько кадров подряд
            # на одном месте) детекция вне гейта есть — создать трек заново по лучу и размеру
            if mode == "predict" and ((c.object_size_m is not None and c.reacq_prior) or self.size_phys) and c.mono_init:
                mono = self._mono_init_candidate(cams, all_cands, cam_ids, c.reacq_prior)
                if mono is not None:
                    i, j, X, cov = mono
                    uv = all_cands[i][j][1]
                    sz = all_cands[i][j][3].size_px
                    pr = self._reacq
                    if pr is not None and pr[0] == frame - 1 and pr[1] == i and np.linalg.norm(uv - pr[2]) < 20 + 3 * sz:
                        self._reacq = (frame, i, uv, pr[3] + 1)
                    else:
                        self._reacq = (frame, i, uv, 1)
                    if self._reacq[3] >= c.reinit_frames:
                        self._init(X, cov, frame, self.x[self.nk:].copy() if self.box else None)
                        mode, used = "init", [cam_ids[i]]
                        info["used_det"] = [(i, all_cands[i][j][0])]
                        info["reinit"] = info["mono_init"] = True
                        self._reacq = None

        if self.x is not None and mode in ("tri", "mono", "mono_size_prior"):
            self._update_omega()
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
            h_px = max(1.0, self._size_px(d))
            cv = self._cv(i, d, cams)
            z = float(cv.R[2] @ self.x[:3] + cv.t.reshape(3)[2])
            if z <= 0:
                continue
            Hm = h_px * z / cv.K[1, 1]
            cid = cam_ids[i]
            self.size_learn_frame[cid] = self.frame
            if cid not in self.size_phys:
                # с априорным размером начинаем с него: одна неудачная триангуляция не испортит оценку
                if self.cfg.object_size_m is not None:
                    self.size_phys[cid] = (1 - a) * self.cfg.object_size_m + a * Hm
                else:
                    self.size_phys[cid] = Hm
            else:
                self.size_phys[cid] = (1 - a) * self.size_phys[cid] + a * Hm

    def _learn_front_offset(self, cams, dets, sig, idx, f) -> None:
        """Смещение «перёд» относительно центра (вдоль курса и по высоте) по точкам, которые дал детектор
        (например, перёд кабины по маске), триангулированным по >= 2 камерам. Копится медиана — причинно."""
        X, _, errs = triangulate_ls([cams[i] for i in idx], [np.asarray(d.point_px, float) for d in dets], list(sig))
        if X is None or errs.max() > self.cfg.pair_reproj_px + self.cfg.pair_reproj_rel * max(d.size_px for d in dets):
            return
        dv = X - self.x[:3]
        self.front_samples.append((float(dv[:2] @ f[:2]), float(dv[2])))
        if len(self.front_samples) > 2000:
            self.front_samples.pop(0)

    def front_offset(self):
        """(вперёд, вверх) в метрах или None, если наблюдений мало."""
        if len(self.front_samples) < self.cfg.front_min_samples:
            return None
        a = np.array(self.front_samples)
        return float(np.median(a[:, 0])), float(np.median(a[:, 1]))

    def _size_px(self, d: Detection) -> float:
        """Размер рамки (px) по выбранному измерению: высота / ширина / среднее геометрическое."""
        w, h = d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]
        if self.cfg.size_dim == "width":
            return float(w)
        if self.cfg.size_dim == "geom":
            return float(np.sqrt(max(w, 0) * max(h, 0)))
        return float(h)

    def _mono_init_candidate(self, cams, all_cands, cam_ids, prior_ok: bool = True):
        """Создание трека по одной камере: точка на луче на дальности f·H/h_px (H — выученный или априорный размер)."""
        c = self.cfg
        best = None
        for i, cl in enumerate(all_cands):
            if cam_ids[i] not in self.size_phys and (c.object_size_m is None or not prior_ok):
                continue  # размер для этой камеры неизвестен — дальность не оценить
            for j, (_, uv, s, d) in enumerate(cl):
                if d.score < c.init_min_score or d.cls in c.box_point_classes:
                    continue
                if best is None or d.score > best[0]:
                    best = (d.score, i, j)
        if best is None:
            return None
        _, i, j = best
        _, uv, s, d = all_cands[i][j]
        cam = self._cv(i, d, cams)
        Hm = self.size_phys.get(cam_ids[i])
        rel = c.size_range_rel_std
        if Hm is None:
            Hm, rel = c.object_size_m, c.object_size_rel_std
        h_px = max(2.0, self._size_px(d))
        zm = cam.K[1, 1] * Hm / h_px
        C, dirs = cam.rays(np.asarray(uv).reshape(1, 2))
        dvec = dirs[0]
        X = C + dvec * zm / max(float(dvec @ cam.R[2]), 1e-6)
        rng = float(np.linalg.norm(X - C))
        sp = s * rng / cam.K[0, 0]
        sr = rel * rng
        cov = np.eye(3) * sp ** 2 + np.outer(dvec, dvec) * (sr ** 2 - sp ** 2)
        return i, j, X, cov

    def _size_measure_box(self, cam, d: Detection):
        c = self.cfg
        if not c.size_from_dims or self.dims_est is None or len(self.dims_est.L_samples) < 10 \
                or np.linalg.norm(self.heading) < 0.5:
            return None
        W, H = cam.size
        b = d.bbox
        if b[0] <= 3 or b[1] <= 3 or b[2] >= W - 3 or b[3] >= H - 3:
            return None  # рамка обрезана краем кадра
        hb = DimsEstimator.box_bbox(cam, self.x[:3], self.heading, self.dims_est.dims)
        if hb is None:
            return None
        s_pred = math.sqrt(max(hb[2] - hb[0], 1e-3) * max(hb[3] - hb[1], 1e-3))
        s_obs = math.sqrt(max(b[2] - b[0], 1e-3) * max(b[3] - b[1], 1e-3))
        zp = float(cam.R[2] @ self.x[:3] + cam.t.reshape(3)[2])
        zm = zp * s_pred / s_obs
        s = c.size_range_rel_std * zm
        varp = float(cam.R[2] @ self.P[:3, :3] @ cam.R[2])
        if (zm - zp) ** 2 / (varp + s * s) > CHI2_2:
            return None
        return [(cam.R[2].copy(), zm - float(cam.t.reshape(3)[2]), s)]

    def _size_measure(self, cam, cid: int, d: Detection):
        """Линейное измерение глубины (ось z камеры) по размеру рамки.

        Если известны размеры объекта (DimsEstimator) и курс — ожидаемая рамка параллелепипеда на прогнозной
        глубине сравнивается с наблюдаемой: z = z_прогн · s_ожид / s_набл (s — среднее геометрическое сторон);
        это учитывает поворот объекта относительно камеры. Иначе — z = f·H/h_px (H выучен для камеры или априорный)."""
        box_z = self._size_measure_box(cam, d)
        if box_z is not None:
            return box_z
        Hm = self.size_phys.get(cid)
        rel = self.cfg.size_range_rel_std
        if Hm is not None and self.cfg.size_rel_growth is not None:
            # размер выучен по этой камере, пока объект был виден >= 2 камерами: сразу после этого он точен
            # (тот же ракурс), дальше ракурс меняется — σ растёт со временем с последнего обучения
            dt = (self.frame - self.size_learn_frame.get(cid, self.frame)) / self.cfg.fps
            rel = min(self.cfg.size_range_rel_std, self.cfg.size_rel_min + self.cfg.size_rel_growth * dt)
        if Hm is None:
            if self.cfg.object_size_m is None:
                return None
            Hm, rel = self.cfg.object_size_m, self.cfg.object_size_rel_std
        h_px = self._size_px(d)
        if h_px < 4:
            return None
        zm = cam.K[1, 1] * Hm / h_px
        s = rel * zm
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
        fo = self.front_offset() if (c.anchor == "front" and c.front_source == "detector_point") else None
        if fo is not None:
            p = p + self.heading * fo[0]
            p[2] += fo[1]
        elif c.anchor != "center" and (self.box or self.dims_est is not None):
            dims = self.x[self.nk:self.nk + 3] if self.box else self.dims_est.dims
            L, Wd, Hd = np.maximum(dims, 0.0)
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


# ---------------------------------------------------------------------------
# Несколько объектов (опция)
# ---------------------------------------------------------------------------

class MultiTracker3D:
    """Несколько независимых треков Tracker3D с исключительным распределением детекций.

    На кадре треки обрабатываются по убыванию «стажа» (числа обновлений): детекции, использованные треком,
    убираются из списка для следующих. Новый трек заводится, только если среди оставшихся детекций есть
    сильная многокамерная гипотеза (как при инициализации одиночного трека). Трек, потерянный дольше
    drop_after_s, удаляется. Основной объект (для trajectory.csv) на кадре k — трек с наибольшим числом
    обновлений к кадру k (выбор причинный). Классы можно ограничить: track_classes — какие детекции
    вообще отдаются трекам."""

    def __init__(self, cfg: TrackerConfig | None = None, max_tracks: int = 3, drop_after_s: float = 10.0):
        self.cfg = cfg or TrackerConfig()
        self.max_tracks = max_tracks
        self.drop_after_s = drop_after_s
        self.tracks: list[tuple[int, Tracker3D]] = []
        self.hits: dict[int, int] = {}
        self._next = 1

    def step(self, frame: int, cams: list, dets: list[list[Detection]], cam_ids: list[int] | None = None) -> dict[int, FrameResult]:
        remaining = [list(d) for d in dets]
        out: dict[int, FrameResult] = {}
        order = sorted(self.tracks, key=lambda t: -self.hits.get(t[0], 0))
        for tid, tr in order:
            if tr.x is None:
                continue
            # потерянный трек не перезапускаем на чужих детекциях — пусть только прогнозирует
            if tr._lost():
                r = tr.step(frame, cams, [[] for _ in cams], cam_ids)
            else:
                r = tr.step(frame, cams, remaining, cam_ids)
            out[tid] = r
            if r.mode in ("tri", "mono", "mono_size_prior", "init"):
                self.hits[tid] = self.hits.get(tid, 0) + 1
            for i, j in sorted(r.info.get("used_det", []), key=lambda t: -t[1]):
                if j < len(remaining[i]):
                    remaining[i].pop(j)
        # новый трек
        if len(self.tracks) < self.max_tracks and any(remaining):
            tr = Tracker3D(self.cfg)
            tr.frame = frame - 1 if frame > 0 else -1
            r = tr.step(frame, cams, remaining, cam_ids)
            if tr.x is not None:
                tid = self._next
                self._next += 1
                self.tracks.append((tid, tr))
                self.hits[tid] = 1
                out[tid] = r
        lim = self.drop_after_s * self.cfg.fps
        self.tracks = [(tid, tr) for tid, tr in self.tracks if frame - tr.last_update_frame <= lim]
        return out

    def primary(self, results: dict[int, FrameResult]) -> int | None:
        if not results:
            return None
        return max(results, key=lambda t: self.hits.get(t, 0))
