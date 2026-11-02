"""Детекция людей и 2D-скелетов (COCO-17) через YOLO-pose."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KP_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
]
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
]
L_HIP, R_HIP, L_ANKLE, R_ANKLE = 11, 12, 15, 16


@dataclass
class Person2D:
    kp: np.ndarray  # (17,2) пиксели
    kc: np.ndarray  # (17,) уверенность
    bbox: np.ndarray  # (4,) x1,y1,x2,y2
    score: float

    def to_dict(self) -> dict:
        return {
            "bbox": [round(float(v), 1) for v in self.bbox],
            "score": round(float(self.score), 3),
            "kp": [[round(float(x), 1), round(float(y), 1), round(float(c), 2)] for (x, y), c in zip(self.kp, self.kc)],
        }


def _auto_device() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


class YoloPoseDetector:
    def __init__(self, model: str = "yolo11n-pose.pt", imgsz: int = 640, conf: float = 0.4, device: str | None = None):
        from ultralytics import YOLO  # тяжёлый импорт — только когда нужен

        self.model = YOLO(model)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device or _auto_device()

    def __call__(self, images: list[np.ndarray]) -> list[list[Person2D]]:
        if not images:
            return []
        results = self.model.predict(images, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
        out: list[list[Person2D]] = []
        for r in results:
            persons: list[Person2D] = []
            if r.keypoints is not None and r.boxes is not None and len(r.boxes) > 0:
                kps = r.keypoints.data.cpu().numpy()  # (n,17,3)
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                for k, b, s in zip(kps, boxes, scores):
                    persons.append(Person2D(k[:, :2].astype(np.float64), k[:, 2].astype(np.float64), b, float(s)))
            out.append(persons)
        return out
