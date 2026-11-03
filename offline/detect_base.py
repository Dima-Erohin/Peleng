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
  * extra — произвольные данные (сохраняются в кэш, если сериализуемы в JSON). Ключевые точки — в
    extra["kp"] = {"имя": [u, v, conf]}; трекер с point="kp:имя" измеряет именно эту точку (например,
    point="kp:pivot" для детектора позы грузовика; понимается и extra["pivot"] = (u, v) + extra["kp_conf"]),
    а при её отсутствии — point_px.
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
        out = [round(float(v), 2) for v in self.bbox] + [round(float(self.point_px[0]), 2), round(float(self.point_px[1]), 2),
                                                         round(float(self.score), 4), self.cls, self.source]
        ex = _jsonable(self.extra)
        if ex:
            out.append(ex)  # 10-й элемент — extra (например, ключевые точки "kp": {"pivot": [u, v, conf]})
        return out

    @classmethod
    def from_list(cls, a: list) -> "Detection":
        return cls(tuple(a[0:4]), (a[4], a[5]), a[6], a[7], a[8] if len(a) > 8 else "",
                   dict(a[9]) if len(a) > 9 and isinstance(a[9], dict) else {})

    def keypoint(self, name: str):
        """Ключевая точка из extra: extra["kp"][name] = [u, v] или [u, v, conf]; иначе None."""
        ex = self.extra or {}
        kp = ex.get("kp") or {}
        v = kp.get(name) if isinstance(kp, dict) else None
        if v is None and isinstance(ex.get(name), (list, tuple)):
            v = ex[name]  # формат TruckPoseDetector: extra["pivot"] = (u, v), extra["kp_conf"]
            if len(v) == 2 and ex.get("kp_conf") is not None:
                v = [v[0], v[1], ex["kp_conf"]]
        if v is None or len(v) < 2:
            return None
        if len(v) > 2 and v[2] is not None and float(v[2]) < 0.3:
            return None
        return float(v[0]), float(v[1])


def _jsonable(x):
    """Оставить только то, что сериализуется в JSON (числа, строки, списки, словари); numpy -> списки."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            j = _jsonable(v)
            if j is not None:
                out[str(k)] = j
        return out
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return np.round(x.astype(float), 3).tolist()
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return round(float(x), 3)
    if isinstance(x, str) or x is None:
        return x
    return None


class Detector:
    """Базовый класс (не обязателен — достаточно утиной типизации)."""

    name = "base"

    def detect(self, cam_id: int, frame_idx: int, image: np.ndarray, camera_model) -> list[Detection]:
        raise NotImplementedError

    def config(self) -> dict:
        """Параметры детектора — пишутся в ключ кэша детекций."""
        return {"name": self.name}
