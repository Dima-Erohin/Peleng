"""3D-трекинг людей: фильтр Калмана (постоянная скорость) + венгерский алгоритм.

Даёт стабильные ID, сглаженную траекторию, скорость и прогноз при кратком перекрытии.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment

from .reconstruct import Observation3D


class Track:
    _ids = itertools.count(1)

    def __init__(self, obs: Observation3D, t: float, meas_std: dict):
        self.id = next(Track._ids)
        self.x = np.zeros(6)  # x y z vx vy vz
        self.x[:3] = obs.pos
        self.P = np.diag([0.3, 0.3, 0.3, 1.0, 1.0, 1.0]) ** 2
        self.t = t
        self.last_update = t
        self.hits = 1
        self.obs = obs
        self.meas_std = meas_std

    def predict(self, t: float, accel_std: float) -> None:
        dt = max(0.0, t - self.t)
        if dt == 0:
            return
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        q = accel_std**2
        G = np.vstack([np.eye(3) * 0.5 * dt**2, np.eye(3) * dt])
        Q = G @ G.T * q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def update(self, obs: Observation3D) -> None:
        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        s = self.meas_std[obs.mode]
        Rm = np.diag([s[0], s[0], s[1]]) ** 2
        y = obs.pos - H @ self.x
        S = H @ self.P @ H.T + Rm
        Kg = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + Kg @ y
        self.P = (np.eye(6) - Kg @ H) @ self.P
        self.last_update = self.t
        self.hits += 1
        self.obs = obs


class Tracker:
    def __init__(self, gate_m: float = 1.0, max_age_s: float = 1.5, min_hits: int = 2, accel_std: float = 3.0):
        self.gate_m = gate_m
        self.max_age_s = max_age_s
        self.min_hits = min_hits
        self.accel_std = accel_std
        self.meas_std = {"tri": (0.05, 0.05), "mono": (0.20, 0.30)}
        self.tracks: list[Track] = []

    def reset(self) -> None:
        self.tracks = []

    def update(self, t: float, observations: list[Observation3D]) -> list[dict]:
        for tr in self.tracks:
            tr.predict(t, self.accel_std)
        matched_obs: set[int] = set()
        if self.tracks and observations:
            C = np.zeros((len(self.tracks), len(observations)))
            for i, tr in enumerate(self.tracks):
                for j, o in enumerate(observations):
                    C[i, j] = np.linalg.norm(tr.x[:2] - o.pos[:2])
            ri, ci = linear_sum_assignment(C)
            for i, j in zip(ri, ci):
                if C[i, j] <= self.gate_m:
                    self.tracks[i].update(observations[j])
                    matched_obs.add(j)
        for j, o in enumerate(observations):
            if j not in matched_obs:
                self.tracks.append(Track(o, t, self.meas_std))
        self.tracks = [tr for tr in self.tracks if t - tr.last_update <= self.max_age_s]

        out = []
        for tr in self.tracks:
            if tr.hits < self.min_hits:
                continue
            predicted = tr.last_update < t
            pos = tr.x[:3]
            vel = tr.x[3:]
            item = {
                "id": tr.id,
                "pos": _r(pos),
                # точка на полу под отфильтрованным центром таза — гладкая (стопы при ходьбе «гуляют»)
                "floor": _r([pos[0], pos[1], 0.0]),
                "vel": _r(vel),
                "speed": round(float(np.linalg.norm(vel[:2])), 3),
                "mode": tr.obs.mode,
                "cams": tr.obs.cams,
                "reproj_err_px": None if np.isnan(tr.obs.reproj_px) or np.isinf(tr.obs.reproj_px) else round(tr.obs.reproj_px, 2),
                "predicted": bool(predicted),
                "kp3d": None,
            }
            if tr.obs.kp3d is not None and not predicted:
                item["kp3d"] = [None if np.isnan(p[0]) else _r(p) for p in tr.obs.kp3d]
            out.append(item)
        return out


def _r(v, nd: int = 3) -> list[float]:
    return [round(float(x), nd) for x in v]
