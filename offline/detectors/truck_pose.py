"""Детектор грузовика YOLO-pose: рамка + keypoint pivot (эталонная точка). Без вычитания фона — камеры могут двигаться.

Веса (best.pt из train_truck_pose.py) не лежат в репозитории: weights=... или переменная окружения TRUCK_WEIGHTS.
Возвращает Detection(cls="truck", bbox=рамка, point_px=pivot (keypoint 0), score=conf рамки,
extra={"pivot":(x,y), "kp_conf":..., "box_center":(x,y)}). Причинно: каждый кадр независим.
"""
from __future__ import annotations

import os

from ..detect_base import Detection, Detector


class TruckPoseDetector(Detector):
    name = "truck_pose"

    def __init__(self, weights: str | None = None, conf: float = 0.1, imgsz: int | None = None, device: str = "cpu",
                 max_det: int = 3, hflip_tta: bool = False):
        from ultralytics import YOLO

        weights = weights or os.environ.get("TRUCK_WEIGHTS")
        if not weights:
            raise ValueError("укажите weights=... или TRUCK_WEIGHTS (best.pt)")
        self.weights, self.conf, self.device, self.max_det, self.tta = weights, conf, device, max_det, hflip_tta
        self.model = YOLO(weights)
        self.imgsz = imgsz or int(self.model.overrides.get("imgsz", 800))

    def config(self) -> dict:
        return {"name": self.name, "weights": str(self.weights), "conf": self.conf, "imgsz": self.imgsz, "tta": self.tta}

    def _run(self, image):
        return self.model.predict(image, imgsz=self.imgsz, conf=self.conf, max_det=self.max_det, verbose=False, device=self.device)[0]

    def detect(self, cam_id, frame_idx, image, camera_model) -> list[Detection]:
        r = self._run(image)
        out = []
        if r.boxes is None or len(r.boxes) == 0:
            return out
        kp = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
        for i, (s, b) in enumerate(zip(r.boxes.conf.tolist(), r.boxes.xyxy.tolist())):
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            if kp is not None and len(kp) > i and len(kp[i]):
                px, py, kc = float(kp[i][0][0]), float(kp[i][0][1]), float(kp[i][0][2])
            else:
                px, py, kc = c[0], c[1], 0.0
            out.append(Detection(tuple(b), (px, py), float(s), "truck", self.name,
                                 {"pivot": (px, py), "kp_conf": kc, "box_center": c}))
        return out
