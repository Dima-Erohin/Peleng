"""Обученный детектор дрона (YOLO11n, дообученный на синтетике из drone_synth.py, см. train_drone.py).

Веса не хранятся в репозитории: путь передаётся параметром weights (по умолчанию переменная окружения DRONE_WEIGHTS).
Инференс на полном кадре в нативном разрешении (imgsz=1920): дрон 8..120 px при уменьшении теряется.
"""
from __future__ import annotations

import os

from ..detect_base import Detection, Detector


class TrainedDrone(Detector):
    name = "drone_yolo"

    def __init__(self, weights: str | None = None, conf: float = 0.25, imgsz: int = 1920, device: str = "cpu"):
        from ultralytics import YOLO

        weights = weights or os.environ.get("DRONE_WEIGHTS")
        if not weights:
            raise ValueError("укажите weights=... или переменную окружения DRONE_WEIGHTS (best.pt из train_drone.py)")
        self.weights, self.conf, self.imgsz, self.device = weights, conf, imgsz, device
        self.model = YOLO(weights)

    def config(self) -> dict:
        return {"name": self.name, "weights": str(self.weights), "conf": self.conf, "imgsz": self.imgsz}

    def detect(self, cam_id, frame_idx, image, camera_model) -> list[Detection]:
        r = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, verbose=False, device=self.device)[0]
        out = []
        for s, b in zip(r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
            out.append(Detection(tuple(b), ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), float(s), "drone", self.name))
        return out
