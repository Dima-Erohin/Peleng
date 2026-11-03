"""Калибровка положения камер цепочкой по парам (без FastAPI — для тестов и скриптов).

Зачем: доска ChArUco читается недалеко (A4 — до ~2–2.5 м, A3 — до ~3–3.5 м), и в большой комнате
или при камерах «спиной» друг к другу её не видят все камеры сразу. Тогда доску переносят:
  * снимок `anchor` (обычно 0) — доска на полу, задаёт мир (как обычная калибровка: начало в центре доски, Z вверх);
    её видит хотя бы одна камера;
  * следующие снимки — доска где угодно (в руках, наклонно, вертикально), её одновременно видят ≥ 2 камеры.

Наблюдение = (снимок k, камера c, углы доски в пикселях + их id). Неизвестные:
  * поза каждой камеры в мире: X_c = R_c (X_w − C_c)  (поворот + центр камеры C_c);
  * поза доски в каждом снимке, кроме опорного: X_w = R_k X_b + t_k  (в опорном снимке — board.R_wb, board.t_wb).
Внутренние параметры камер фиксированы (из реестра, под размер кадра наблюдения).

Решение:
  1. Начальное приближение — «чистая цепочка»: поза доски в каждой камере по solvePnP, затем кратчайшие пути
     (Дейкстра) в двудольном графе «снимки — камеры» от опорного снимка; вес ребра — грубая оценка ошибки позы
     (дальше, меньше углов, больше ошибка репроекции — хуже). Так каждая камера привязывается по лучшему пути.
  2. Совместное уточнение (bundle adjustment): scipy.optimize.least_squares по ошибке репроекции всех углов во всех
     наблюдениях (пиксели), робастная функция потерь soft_l1. Наблюдения-выбросы (доска сдвинулась, ложная детекция)
     отбрасываются по одному, пока худшее не станет приемлемым.
  3. Неопределённость центров камер — грубо, из якобиана: cov = s² (JᵀJ)⁻¹, s² — остаточная дисперсия (px²).
     Ошибку оптики (K) она не учитывает — реальная ошибка обычно больше; проверять — расширением «Точность».
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .calibration import Board, BoardDetection, _not_collinear
from .geometry import CameraModel

MIN_CORNERS = 6


@dataclass
class Observation:
    snap: int
    cam: str
    corners: np.ndarray  # (N,2) пиксели кадра размера size
    ids: np.ndarray  # (N,) id углов ChArUco
    K: np.ndarray
    dist: np.ndarray
    size: tuple[int, int]  # (w, h) кадра, в котором нашли углы

    def model(self) -> CameraModel:
        return CameraModel(self.K, self.dist, tuple(self.size))


@dataclass
class ChainResult:
    ok: bool
    message: str = ""
    cams: dict = field(default_factory=dict)  # c -> {"R", "t", "center", "reproj_px", ...}
    snaps: dict = field(default_factory=dict)  # k -> {"R_wb", "t_wb", "reproj_px", "cams"}
    chain: dict = field(default_factory=dict)  # c -> {"R", "t"} — начальное приближение (чистая цепочка)
    unlinked: list = field(default_factory=list)
    dropped: list = field(default_factory=list)  # [{"snap", "cam", "reproj_px", "reason"}]
    warnings: list = field(default_factory=list)
    reproj_px: float = float("nan")  # RMS по всем углам после уточнения
    chain_reproj_px: float = float("nan")  # RMS для начального приближения

    def summary(self) -> dict:
        """Для JSON: без матриц поворота."""
        r = lambda v, d=4: None if v is None or not math.isfinite(v) else round(float(v), d)  # noqa: E731
        cams = {}
        for c, x in self.cams.items():
            cams[c] = {
                "center": np.round(x["center"], 4).tolist(),
                "reproj_px": r(x["reproj_px"], 3),
                "points": x["points"],
                "snapshots": x["snapshots"],
                "hops": x["hops"],
                "via": x.get("via"),
                "center_std_m": None if x["center_std_m"] is None else np.round(x["center_std_m"], 4).tolist(),
                "center_std_norm_m": r(x["center_std_norm_m"]),
                "refine_shift_m": r(x["refine_shift_m"]),
            }
        snaps = {
            str(k): {"reproj_px": r(x["reproj_px"], 3), "cams": x["cams"], "corners_world": np.round(x["corners_world"], 4).tolist()}
            for k, x in self.snaps.items()
        }
        return {
            "ok": self.ok,
            "message": self.message,
            "cams": cams,
            "snaps": snaps,
            "unlinked": self.unlinked,
            "dropped": self.dropped,
            "warnings": self.warnings,
            "reproj_px": r(self.reproj_px, 3),
            "chain_reproj_px": r(self.chain_reproj_px, 3),
        }


# ---------------------------------------------------------------------------
# Поза доски в одной камере
# ---------------------------------------------------------------------------

def observation_ok(obs: Observation, board: Board) -> bool:
    if obs.corners is None or len(obs.ids) < MIN_CORNERS:
        return False
    return _not_collinear(BoardDetection(obs.corners, obs.ids, board.all_corners[obs.ids]))


def pnp(board: Board, obs: Observation) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Поза доски в камере (X_c = R X_b + t), ошибка репроекции (px, средняя) и неоднозначность.

    У плоской мишени бывает два близких решения (особенно издалека и «в лоб») — IPPE даёт оба, берём лучшее.
    Неоднозначность = ошибка лучшего / ошибка второго (≈ 1 — решения почти равноправны, поза ненадёжна)."""
    obj = board.all_corners[obs.ids].astype(np.float64)
    img = obs.corners.astype(np.float64)
    try:
        n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, obs.K, obs.dist, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None
    sols = []
    for rv, tv in zip(rvecs[:n], tvecs[:n]):
        rv, tv = cv2.solvePnPRefineLM(obj, img, obs.K, obs.dist, rv, tv)
        proj, _ = cv2.projectPoints(obj, rv, tv, obs.K, obs.dist)
        err = float(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1).mean())
        R, _ = cv2.Rodrigues(rv)
        t = tv.reshape(3)
        if (obj @ R.T + t)[:, 2].min() <= 0:
            continue
        sols.append((err, R, t))
    if not sols:
        return None
    sols.sort(key=lambda x: x[0])
    err, R, t = sols[0]
    amb = (err + 0.05) / (sols[1][0] + 0.05) if len(sols) > 1 else 0.0
    return R, t, err, float(amb)


def edge_cost(obs: Observation, t_cb: np.ndarray, err_px: float, amb: float = 0.0) -> float:
    """Грубая «ошибка» позы доски по одному виду: растёт с расстоянием, ошибкой и неоднозначностью, падает с числом углов."""
    depth = float(np.linalg.norm(t_cb))
    f = float((obs.K[0, 0] + obs.K[1, 1]) / 2)
    return depth * depth / f * (0.3 + err_px) / math.sqrt(len(obs.ids)) * (1.0 + 4.0 * amb ** 2)


# ---------------------------------------------------------------------------
# Граф «снимки — камеры»
# ---------------------------------------------------------------------------

def graph_summary(observations: list[Observation], all_cams: list[str] | None = None, anchor: int = 0,
                  prior_centers: dict[str, np.ndarray] | None = None) -> dict:
    """Что с чем связано: привязанные к миру камеры, число общих снимков у пар, подсказки, куда нести доску."""
    cams = sorted(set(all_cams or []) | {o.cam for o in observations})
    by_snap: dict[int, set[str]] = {}
    for o in observations:
        by_snap.setdefault(o.snap, set()).add(o.cam)
    # обход в ширину по двудольному графу от опорного снимка
    hops: dict[str, int] = {}
    if anchor in by_snap:
        frontier = set(by_snap[anchor])
        for c in frontier:
            hops[c] = 1
        seen_snaps = {anchor}
        level = 1
        while frontier:
            level += 1
            nxt = set()
            for k, cs in by_snap.items():
                if k in seen_snaps or not (cs & frontier):
                    continue
                seen_snaps.add(k)
                for c in cs:
                    if c not in hops:
                        hops[c] = level
                        nxt.add(c)
            frontier = nxt
    pairs: dict[tuple[str, str], int] = {}
    for cs in by_snap.values():
        cs = sorted(cs)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                pairs[(cs[i], cs[j])] = pairs.get((cs[i], cs[j]), 0) + 1
    n_snaps = {c: sum(c in cs for cs in by_snap.values()) for c in cams}

    hints = []
    linked = [c for c in cams if c in hops]
    unlinked = [c for c in cams if c not in hops]
    if anchor not in by_snap:
        hints.append("Положите доску на пол так, чтобы её видела хотя бы одна камера, и нажмите «Снимок» — это задаст систему координат.")
    else:
        for c in unlinked:
            partner = None
            if prior_centers and c in prior_centers:
                near = [(float(np.linalg.norm(prior_centers[c] - prior_centers[x])), x) for x in linked if x in prior_centers]
                if near:
                    partner = min(near)[1]
            if partner is None:
                # без прежней калибровки — с той привязанной, с которой у неё уже есть общие снимки, иначе с ближайшей к миру
                shared = [(pairs.get(tuple(sorted((c, x))), 0), -hops[x], x) for x in linked]
                partner = max(shared)[2] if shared else None
            if partner:
                hints.append(f"Переместите доску между {c} и {partner} (чтобы её видели обе камеры) и нажмите «Снимок».")
            else:
                hints.append(f"Камера {c} пока не связана: покажите доску ей и какой-нибудь привязанной камере одновременно.")
        # привязанные, но лишь одним общим снимком — ненадёжно
        for c in linked:
            if hops[c] == 1:
                continue
            links = [(pairs.get(tuple(sorted((c, x))), 0), x) for x in linked if x != c and hops[x] < hops[c]]
            best = max(links) if links else (0, None)
            if best[0] == 1:
                hints.append(f"Ещё 1–2 снимка между {c} и {best[1]} (доска в другом месте/наклоне) — надёжнее.")
        if not unlinked and len(linked) >= 1:
            hints.append("Все камеры связаны — можно «Применить» (или добавить снимков для точности).")
    return {
        "cams": {c: {"linked": c in hops, "hops": hops.get(c), "snapshots": n_snaps[c]} for c in cams},
        "pairs": [{"a": a, "b": b, "n": n} for (a, b), n in sorted(pairs.items())],
        "linked": linked,
        "unlinked": unlinked,
        "hints": hints,
    }


# ---------------------------------------------------------------------------
# Начальное приближение: кратчайшие пути
# ---------------------------------------------------------------------------

def initial_chain(board: Board, observations: list[Observation], anchor: int = 0, pnp_cache: dict | None = None):
    """Чистая цепочка. Возвращает (cams: c -> (R_cw, t_cw), snaps: k -> (R_wb, t_wb), info: c -> {"hops", "via"})."""
    poses = pnp_cache if pnp_cache is not None else {}
    adj: dict[tuple, list] = {}
    for o in observations:
        key = (o.snap, o.cam)
        if key not in poses:
            poses[key] = pnp(board, o)
        p = poses[key]
        if p is None:
            continue
        w = edge_cost(o, p[1], p[2], p[3])
        adj.setdefault(("s", o.snap), []).append((w, ("c", o.cam), key))
        adj.setdefault(("c", o.cam), []).append((w, ("s", o.snap), key))
    cams: dict[str, tuple] = {}
    snaps: dict[int, tuple] = {}
    info: dict[str, dict] = {}
    start = ("s", anchor)
    if start not in adj:
        return cams, snaps, info
    snaps[anchor] = (board.R_wb.copy(), board.t_wb.copy())
    dist = {start: 0.0}
    depth = {start: 0}
    heap = [(0.0, 0, start, None, None)]
    done = set()
    tie = 0
    while heap:
        d, _, node, parent, key = heapq.heappop(heap)
        if node in done:
            continue
        done.add(node)
        if parent is not None:
            R_cb, t_cb = poses[key][:2]
            if node[0] == "c":
                R_wb, t_wb = snaps[parent[1]]
                R_cw = R_cb @ R_wb.T
                cams[node[1]] = (R_cw, t_cb - R_cw @ t_wb)
                info[node[1]] = {"hops": (depth[node] + 1) // 2, "via": parent[1]}
            else:
                R_cw, t_cw = cams[parent[1]]
                snaps[node[1]] = (R_cw.T @ R_cb, R_cw.T @ (t_cb - t_cw))
        for w, nb, k in adj.get(node, []):
            if nb in done:
                continue
            nd = d + w
            if nd < dist.get(nb, float("inf")):
                dist[nb] = nd
                depth[nb] = depth[node] + 1
                tie += 1
                heapq.heappush(heap, (nd, tie, nb, node, k))
    return cams, snaps, info


# ---------------------------------------------------------------------------
# Совместное уточнение
# ---------------------------------------------------------------------------

class _Problem:
    """Невязки: по 2 на угол доски в наблюдении (пиксели) + по 4 на снимок «доска на полу» (углы доски на высоте пола)."""

    def __init__(self, board: Board, observations: list[Observation], cam_ids: list[str], snap_ids: list[int], anchor: int,
                 floor_snaps=(), floor_w: float = 500.0):
        self.cam_ids = cam_ids
        self.snap_ids = snap_ids  # без опорного
        ci = {c: i for i, c in enumerate(cam_ids)}
        si = {k: i for i, k in enumerate(snap_ids)}
        n_s = len(snap_ids)
        rows_c, rows_s, Xb, uv, f, obs_idx = [], [], [], [], [], []
        for j, o in enumerate(observations):
            n = len(o.ids)
            rows_c.append(np.full(n, ci[o.cam]))
            rows_s.append(np.full(n, n_s if o.snap == anchor else si[o.snap]))  # n_s — опорный снимок
            Xb.append(board.all_corners[o.ids])
            uv.append(o.model().undistort(o.corners))
            f.append(np.full(n, (o.K[0, 0] + o.K[1, 1]) / 2))
            obs_idx.append(np.full(n, j))
        self.ci = np.concatenate(rows_c)
        self.si = np.concatenate(rows_s)
        self.Xb = np.vstack(Xb)
        self.uv = np.vstack(uv)
        self.f = np.concatenate(f)
        self.obs_idx = np.concatenate(obs_idx)
        self.n_c, self.n_s = len(cam_ids), n_s
        self.R0 = board.R_wb
        self.t0 = board.t_wb
        # снимки «доска на полу» (кроме опорного): 4 угла доски должны лежать на высоте пола — это фиксирует
        # наклон пола и высоту там, где опорный снимок видно плохо. floor_w — пикселей на метр (500: 2 мм ≈ 1 px)
        w, h = board.size_m
        self.quad = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], dtype=np.float64)
        self.floor_idx = np.array([si[k] for k in floor_snaps if k in si], dtype=int)
        self.floor_z = float(board.t_wb[2])
        self.floor_w = float(floor_w)
        self.n_reproj = 2 * len(self.ci)
        # столбцы якобиана для каждой невязки (−1 — не зависит)
        cam_col = np.repeat(6 * self.ci, 2)
        snap_col = np.repeat(np.where(self.si < n_s, 6 * self.n_c + 6 * self.si, -1), 2)
        if len(self.floor_idx):
            cam_col = np.concatenate([cam_col, np.full(4 * len(self.floor_idx), -1)])
            snap_col = np.concatenate([snap_col, np.repeat(6 * self.n_c + 6 * self.floor_idx, 4)])
        self.cam_col, self.snap_col = cam_col, snap_col

    def pack(self, cams: dict, snaps: dict) -> np.ndarray:
        x = []
        for c in self.cam_ids:
            R, t = cams[c]
            x += [*Rotation.from_matrix(R).as_rotvec(), *(-R.T @ t)]
        for k in self.snap_ids:
            R, t = snaps[k]
            x += [*Rotation.from_matrix(R).as_rotvec(), *t]
        return np.asarray(x, dtype=np.float64)

    def unpack(self, x: np.ndarray):
        pc = x[: 6 * self.n_c].reshape(-1, 6)
        ps = x[6 * self.n_c:].reshape(-1, 6)
        Rc = Rotation.from_rotvec(pc[:, :3]).as_matrix().reshape(-1, 3, 3)
        Cc = pc[:, 3:]
        Rs = np.concatenate([Rotation.from_rotvec(ps[:, :3]).as_matrix().reshape(-1, 3, 3), self.R0[None]]) if len(ps) else self.R0[None]
        ts = np.concatenate([ps[:, 3:], self.t0[None]]) if len(ps) else self.t0[None]
        return Rc, Cc, Rs, ts

    def residuals(self, x: np.ndarray) -> np.ndarray:
        Rc, Cc, Rs, ts = self.unpack(x)
        Xw = np.einsum("nij,nj->ni", Rs[self.si], self.Xb) + ts[self.si]
        Xc = np.einsum("nij,nj->ni", Rc[self.ci], Xw - Cc[self.ci])
        z = np.maximum(Xc[:, 2], 1e-6)
        r = ((Xc[:, :2] / z[:, None] - self.uv) * self.f[:, None]).reshape(-1)
        if len(self.floor_idx):
            zf = np.einsum("kj,mj->km", Rs[self.floor_idx][:, 2, :], self.quad) + ts[self.floor_idx][:, 2:3]
            r = np.concatenate([r, (zf - self.floor_z).reshape(-1) * self.floor_w])
        return r

    def jacobian(self, x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Якобиан конечными разностями по блокам: невязка зависит только от своей камеры и своего снимка,
        поэтому d-й параметр сдвигается сразу у всех камер (или у всех снимков) — 12 вычислений вместо числа параметров."""
        r0 = self.residuals(x)
        J = np.zeros((len(r0), len(x)))
        rows = np.arange(len(r0))
        hc, hs = self.cam_col >= 0, self.snap_col >= 0
        for d in range(6):
            xc = x.copy()
            xc[d: 6 * self.n_c: 6] += eps
            J[rows[hc], self.cam_col[hc] + d] = ((self.residuals(xc) - r0) / eps)[hc]
            if self.n_s:
                xs = x.copy()
                xs[6 * self.n_c + d:: 6] += eps
                J[rows[hs], self.snap_col[hs] + d] = ((self.residuals(xs) - r0) / eps)[hs]
        return J


def _rms(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(v ** 2))) if len(v) else float("nan")


def bundle_adjust(board: Board, observations: list[Observation], cams0: dict, snaps0: dict, anchor: int = 0,
                  loss: str = "soft_l1", f_scale: float = 1.0, floor_snaps=(), floor_w: float = 500.0):
    """Совместное уточнение. Наблюдения — только те, у которых камера и снимок есть в начальном приближении.
    floor_snaps — снимки, где доска тоже лежала на полу (кроме опорного): мягкое ограничение «углы доски на высоте пола».

    Возвращает (cams, snaps, per_obs_rms (px, по наблюдениям), center_cov: c -> 3×3, rms_all)."""
    cam_ids = sorted(cams0)
    snap_ids = sorted(k for k in snaps0 if k != anchor)
    prob = _Problem(board, observations, cam_ids, snap_ids, anchor, floor_snaps, floor_w)
    x0 = prob.pack(cams0, snaps0)
    res = least_squares(prob.residuals, x0, jac=prob.jacobian, method="trf", loss=loss, f_scale=f_scale, x_scale="jac", max_nfev=100)
    x = res.x
    Rc, Cc, Rs, ts = prob.unpack(x)
    cams = {c: (Rc[i], -Rc[i] @ Cc[i]) for i, c in enumerate(cam_ids)}
    snaps = {k: (Rs[i], ts[i]) for i, k in enumerate(snap_ids)}
    snaps[anchor] = (board.R_wb.copy(), board.t_wb.copy())
    r = prob.residuals(x)[: prob.n_reproj].reshape(-1, 2)  # только пиксели, без невязок пола
    per_obs = np.array([_rms(r[prob.obs_idx == j]) for j in range(len(observations))])
    # грубая ковариация: s² (JᵀJ)⁻¹ (J — якобиан с учётом робастной функции потерь)
    cov_c: dict[str, np.ndarray] = {}
    try:
        J = np.asarray(res.jac)
        m, n = J.shape
        dof = max(m - n, 1)
        s2 = float(np.sum(r ** 2)) / dof
        cov = np.linalg.pinv(J.T @ J) * s2
        for i, c in enumerate(cam_ids):
            cov_c[c] = cov[6 * i + 3: 6 * i + 6, 6 * i + 3: 6 * i + 6]
    except (np.linalg.LinAlgError, ValueError, MemoryError):
        pass
    return cams, snaps, per_obs, cov_c, _rms(r)


def _chain_rms(board: Board, observations: list[Observation], cams: dict, snaps: dict, anchor: int) -> float:
    use = [o for o in observations if o.cam in cams and o.snap in snaps]
    if not use:
        return float("nan")
    prob = _Problem(board, use, sorted(cams), sorted(k for k in snaps if k != anchor), anchor)
    return _rms(prob.residuals(prob.pack(cams, snaps)))


# ---------------------------------------------------------------------------
# Полное решение
# ---------------------------------------------------------------------------

def solve(board: Board, observations: list[Observation], anchor: int = 0, refine: bool = True, floor_snaps=(),
          outlier_px: float = 3.0, max_drop: int = 5, warn_reproj_px: float = 1.0, warn_std_m: float = 0.03,
          warn_hops: int = 3) -> ChainResult:
    """Позы всех камер, связанных с опорным снимком (см. описание модуля).

    floor_snaps — номера снимков, где доска тоже лежала на полу (в другом месте): уточняют наклон пола и высоту."""
    all_cams = sorted({o.cam for o in observations})
    obs = []
    dropped = []
    for o in observations:
        if observation_ok(o, board):
            obs.append(o)
        else:
            dropped.append({"snap": o.snap, "cam": o.cam, "reproj_px": None, "reason": "мало углов или они на одной линии"})
    if not any(o.snap == anchor for o in obs):
        return ChainResult(False, "нет опорного снимка (доска на полу, видимая хотя бы одной камерой)",
                           unlinked=all_cams, dropped=dropped)
    cache: dict = {}
    for o in obs:
        cache[(o.snap, o.cam)] = pnp(board, o)
        if cache[(o.snap, o.cam)] is None:
            dropped.append({"snap": o.snap, "cam": o.cam, "reproj_px": None, "reason": "solvePnP не сошёлся"})
    obs = [o for o in obs if cache[(o.snap, o.cam)] is not None]

    for it in range(max_drop + 1):
        cams0, snaps0, info = initial_chain(board, obs, anchor, cache)
        if not cams0:
            return ChainResult(False, "опорный снимок не дал ни одной камеры", unlinked=all_cams, dropped=dropped)
        use = [o for o in obs if o.cam in cams0 and o.snap in snaps0]
        chain_rms = _chain_rms(board, use, cams0, snaps0, anchor)
        if refine:
            cams, snaps, per_obs, cov_c, rms = bundle_adjust(board, use, cams0, snaps0, anchor, floor_snaps=floor_snaps)
        else:
            cams, snaps, cov_c = cams0, snaps0, {}
            prob = _Problem(board, use, sorted(cams0), sorted(k for k in snaps0 if k != anchor), anchor)
            r = prob.residuals(prob.pack(cams0, snaps0)).reshape(-1, 2)
            per_obs = np.array([_rms(r[prob.obs_idx == j]) for j in range(len(use))])
            rms = _rms(r)
        # выбросы: худшее наблюдение заметно хуже остальных — убрать и пересчитать
        if not refine or len(use) < 2 or it == max_drop:
            break
        j = int(np.argmax(per_obs))
        thr = max(outlier_px, 4.0 * float(np.median(per_obs)))
        if per_obs[j] <= thr:
            break
        bad = use[j]
        dropped.append({"snap": bad.snap, "cam": bad.cam, "reproj_px": round(float(per_obs[j]), 2),
                        "reason": "выброс (доска сдвинулась или углы найдены неверно)"})
        obs = [o for o in obs if o is not bad]

    # --- итоги
    res = ChainResult(True, "ok", reproj_px=rms, chain_reproj_px=chain_rms, dropped=dropped)
    res.unlinked = [c for c in all_cams if c not in cams]
    per_cam_rows: dict[str, list] = {}
    for o, e in zip(use, per_obs):
        per_cam_rows.setdefault(o.cam, []).append((o, e))
    for c, (R, t) in cams.items():
        rows = per_cam_rows.get(c, [])
        n_pts = sum(len(o.ids) for o, _ in rows)
        rms_c = math.sqrt(sum(e ** 2 * len(o.ids) for o, e in rows) / n_pts) if n_pts else float("nan")
        C = -R.T @ t
        R0, t0 = cams0[c]
        cov = cov_c.get(c)
        std = np.sqrt(np.clip(np.diag(cov), 0, None)) if cov is not None else None
        res.cams[c] = {
            "R": R, "t": t, "center": C, "reproj_px": rms_c, "points": n_pts, "snapshots": sorted({o.snap for o, _ in rows}),
            "hops": info[c]["hops"], "via": info[c]["via"],
            "center_std_m": std, "center_std_norm_m": float(np.linalg.norm(std)) if std is not None else None,
            "refine_shift_m": float(np.linalg.norm(C - (-R0.T @ t0))),
        }
        res.chain[c] = {"R": R0, "t": t0}
    w, h = board.size_m
    quad = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], dtype=np.float64)
    for k, (R, t) in snaps.items():
        rows = [(o, e) for o, e in zip(use, per_obs) if o.snap == k]
        n_pts = sum(len(o.ids) for o, _ in rows)
        res.snaps[k] = {
            "R_wb": R, "t_wb": t, "cams": sorted(o.cam for o, _ in rows),
            "reproj_px": math.sqrt(sum(e ** 2 * len(o.ids) for o, e in rows) / n_pts) if n_pts else float("nan"),
            "corners_world": quad @ R.T + t,
        }

    # --- предупреждения
    g = graph_summary(use, anchor=anchor)
    pair_n = {(p["a"], p["b"]): p["n"] for p in g["pairs"]}
    for c, x in sorted(res.cams.items()):
        if x["hops"] >= warn_hops:
            res.warnings.append(f"{c}: {x['hops']}-е звено цепочки — ошибка накапливается, проверьте точность (блок «Точность»)")
        if x["hops"] > 1 and x["via"] is not None:
            # с какими уже привязанными камерами у этой есть общие снимки
            links = max((n for (a, b), n in pair_n.items() if c in (a, b) and res.cams.get(b if a == c else a, {}).get("hops", 99) < x["hops"]), default=0)
            if links < 2:
                res.warnings.append(f"{c}: привязана всего одним общим снимком — добавьте ещё 1–2")
        if math.isfinite(x["reproj_px"]) and x["reproj_px"] > warn_reproj_px:
            res.warnings.append(f"{c}: ошибка репроекции {x['reproj_px']:.2f} px (> {warn_reproj_px:g}) — проверьте оптику и неподвижность доски")
        if x["center_std_norm_m"] is not None and x["center_std_norm_m"] > warn_std_m:
            res.warnings.append(f"{c}: неопределённость положения ±{x['center_std_norm_m'] * 100:.1f} см — доску ближе к камере или больше снимков")
    for c in res.unlinked:
        res.warnings.append(f"{c}: не связана с опорным снимком — её положение не меняется")
    if dropped:
        res.warnings.append(f"отброшено наблюдений: {len(dropped)}")
    return res
