"""Общий интерфейс детектора для офлайн-конвейера.

Любой детектор (базовые — offline/basic_detectors.py, для летающего объекта — offline/detectors/)
реализует класс с методом

    detect(cam_id: int, frame_idx: int, image: np.ndarray (BGR), camera_model: CameraModel) -> list[Detection]

Правила:
  * ПРИЧИННОСТЬ: при вызове для кадра k детектор может использовать только кадры <= k
    (свои внутренние состояния — фон, история — копятся из предыдущих вызовов). Вызовы идут строго
    по возрастанию frame_idx для каждой камеры; кадры разных камер одного момента приходят подряд.
  * Координаты — пиксели исходного кадра (не уменьшенного).
  * point_px — точка объекта, которую надо триангулировать (по умолчанию центр рамки); детектор может
    задать свою (например, центр масс маски).
  * score в [0, 1]; cls — строка ("truck", "drone", "motion", ...); source — имя детектора.
  * camera_model — модель камеры на этот кадр (рабочая система координат, см. ue_data.py); может
    пригодиться, например, для компенсации поворота камеры.

Регистрация: make_detector(name, **kw) в basic_detectors.py знает «bg», «yolo», «yolo+bg»;
внешний детектор можно указать как «модуль:Класс» (например offline.detectors.drone:DroneDetector) —
класс создаётся без аргументов или с **kw.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (пиксели)
    point_px: tuple[float, float]  # точка объекта (пиксели)
    score: float
    cls: str = "object"
    source: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def size_px(self) -> float:
        """Характерный размер рамки (среднее геометрическое сторон)."""
        w = max(1.0, self.bbox[2] - self.bbox[0])
        h = max(1.0, self.bbox[3] - self.bbox[1])
        return float(np.sqrt(w * h))

    def to_list(self) -> list:
        return [round(float(v), 2) for v in self.bbox] + [round(float(self.point_px[0]), 2), round(float(self.point_px[1]), 2),
                                                          round(float(self.score), 4), self.cls, self.source]

    @classmethod
    def from_list(cls, a: list) -> "Detection":
        return cls(tuple(a[0:4]), (a[4], a[5]), a[6], a[7], a[8] if len(a) > 8 else "")


class Detector:
    """Базовый класс (не обязателен — достаточно утиной типизации)."""

    name = "base"

    def detect(self, cam_id: int, frame_idx: int, image: np.ndarray, camera_model) -> list[Detection]:
        raise NotImplementedError

    def config(self) -> dict:
        """Параметры детектора — пишутся в ключ кэша детекций."""
        return {"name": self.name}
