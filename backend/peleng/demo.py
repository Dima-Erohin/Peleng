"""Демо-режим без камер: виртуальная комната, виртуальные камеры, люди ходят по кругу.

Генерируются «детекции» YOLO (проекции синтетического скелета + шум), дальше работает
настоящий конвейер: сопоставление, триангуляция, трекинг. Нужен:
  * Максу — чтобы делать фронтенд без железа;
  * Диме — чтобы проверять алгоритмы и точность (известна истинная позиция).
"""
from __future__ import annotations

import math
import threading
import time

import cv2
import numpy as np

from .geometry import CameraModel, approx_K, look_at
from .pose import SKELETON, Person2D
from .sources import Source

# Скелет в системе тела: x — вперёд, y — влево, z — вверх (метры)
_BASE = {
    0: (0.10, 0.00, 1.62), 1: (0.09, 0.035, 1.66), 2: (0.09, -0.035, 1.66), 3: (0.02, 0.08, 1.63), 4: (0.02, -0.08, 1.63),
    5: (0.0, 0.20, 1.44), 6: (0.0, -0.20, 1.44), 7: (0.0, 0.24, 1.15), 8: (0.0, -0.24, 1.15),
    9: (0.05, 0.26, 0.88), 10: (0.05, -0.26, 0.88), 11: (0.0, 0.11, 0.95), 12: (0.0, -0.11, 0.95),
    13: (0.03, 0.11, 0.52), 14: (0.03, -0.11, 0.52), 15: (0.0, 0.11, 0.08), 16: (0.0, -0.11, 0.08),
}


class DemoWorld:
    def __init__(self, room=(5.0, 4.0, 2.7), n_cams: int = 3, n_people: int = 2, size=(960, 540), hfov=68.0):
        self.room = room
        W, D, H = room
        corners = [(-W / 2 + 0.1, -D / 2 + 0.1), (W / 2 - 0.1, -D / 2 + 0.1), (W / 2 - 0.1, D / 2 - 0.1), (-W / 2 + 0.1, D / 2 - 0.1)]
        self.cams: dict[str, CameraModel] = {}
        for i in range(n_cams):
            cx, cy = corners[i % 4]
            R, t = look_at(np.array([cx, cy, 2.2]), np.array([0.0, 0.0, 0.8]))
            self.cams[f"demo{i + 1}"] = CameraModel(approx_K(*size, hfov), np.zeros(5), size, R, t, meta={"intrinsics": "demo"})
        self.n_people = n_people
        self.t0 = time.time()

    def people(self, t: float) -> list[tuple[np.ndarray, np.ndarray]]:
        """[(центр на полу (3,), скелет (17,3))] в момент t."""
        tt = t - self.t0
        out = []
        for k in range(self.n_people):
            if k == 0:
                w = 2 * math.pi / 14
                p = np.array([1.5 * math.cos(w * tt), 1.0 * math.sin(w * tt), 0.0])
                v = np.array([-1.5 * w * math.sin(w * tt), 1.0 * w * math.cos(w * tt)])
            else:
                w = 2 * math.pi / 20
                p = np.array([1.2 * math.sin(w * tt + k), 0.8 * math.sin(2 * w * tt + k), 0.0])
                v = np.array([1.2 * w * math.cos(w * tt + k), 1.6 * w * math.cos(2 * w * tt + k)])
            h = v / (np.linalg.norm(v) + 1e-9)
            left = np.array([-h[1], h[0]])
            phase = 2 * math.pi * 1.8 * tt + k
            sk = np.zeros((17, 3))
            for j, (bx, by, bz) in _BASE.items():
                swing = 0.0
                if j in (13, 15):
                    swing = 0.25 * math.sin(phase) * (1 if j == 15 else 0.5)
                elif j in (14, 16):
                    swing = -0.25 * math.sin(phase) * (1 if j == 16 else 0.5)
                elif j in (9, 7):
                    swing = -0.15 * math.sin(phase)
                elif j in (10, 8):
                    swing = 0.15 * math.sin(phase)
                xy = p[:2] + h * (bx + swing) + left * by
                sk[j] = [xy[0], xy[1], bz]
            out.append((p, sk))
        return out


class DemoSource(Source):
    kind = "demo"

    def __init__(self, cam_id: str, world: DemoWorld, fps: float = 15.0):
        super().__init__(cam_id, cam_id)
        self.world = world
        self.cam = world.cams[cam_id]
        self.fps_target = fps
        self._running = False
        self._grid = self._grid_lines()

    def _grid_lines(self):
        W, D, _ = self.world.room
        lines = []
        for x in np.arange(-W / 2, W / 2 + 1e-6, 0.5):
            lines.append(([x, -D / 2, 0], [x, D / 2, 0]))
        for y in np.arange(-D / 2, D / 2 + 1e-6, 0.5):
            lines.append(([-W / 2, y, 0], [W / 2, y, 0]))
        return lines

    def render(self, t: float) -> np.ndarray:
        w, h = self.cam.size
        img = np.full((h, w, 3), 40, np.uint8)
        for a, b in self._grid:
            seg = np.linspace(a, b, 20)
            if (self.cam.depth(seg) <= 0.1).any():
                continue
            px = self.cam.project(seg).astype(np.int32)
            cv2.polylines(img, [px], False, (90, 90, 90), 1, cv2.LINE_AA)
        for _, sk in self.world.people(t):
            if (self.cam.depth(sk) <= 0.1).any():
                continue
            px = self.cam.project(sk)
            for i, j in SKELETON:
                cv2.line(img, tuple(px[i].astype(int)), tuple(px[j].astype(int)), (220, 220, 220), 6, cv2.LINE_AA)
        cv2.putText(img, f"{self.id} (demo)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        return img

    def _loop(self):
        while self._running:
            t = time.time()
            self._put(self.render(t), t)
            time.sleep(max(0.0, 1.0 / self.fps_target - (time.time() - t)))

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name=f"demo-{self.id}").start()

    def stop(self):
        self._running = False


class DemoDetector:
    """Вместо YOLO: проецирует истинный скелет в камеру с шумом."""

    def __init__(self, world: DemoWorld, noise_px: float = 1.5, drop_prob: float = 0.05, seed: int = 0):
        self.world = world
        self.noise_px = noise_px
        self.drop_prob = drop_prob
        self.rng = np.random.default_rng(seed)

    def __call__(self, batch) -> list[list[Person2D]]:
        out = []
        for cam_id, frame in batch:
            cam = self.world.cams[cam_id]
            w, h = cam.size
            persons = []
            for _, sk in self.world.people(frame.ts):
                if (cam.depth(sk) <= 0.1).any():
                    continue
                px = cam.project(sk) + self.rng.normal(0, self.noise_px, (17, 2))
                inside = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
                if inside.sum() < 5:
                    continue
                kc = np.where(inside, 0.9, 0.0)
                kc[self.rng.random(17) < self.drop_prob] = 0.1
                vis = px[inside]
                bbox = np.array([vis[:, 0].min(), vis[:, 1].min(), vis[:, 0].max(), vis[:, 1].max()])
                persons.append(Person2D(px, kc, bbox, 0.9))
            out.append(persons)
        return out
