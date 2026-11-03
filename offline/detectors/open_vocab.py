"""Open-vocabulary детектор дрона (YOLOE, текстовые промпты) на тайлах с увеличением.

Находка: на полном кадре 1920x1080 YOLOE дрон (≈73 px) не видит (даже при imgsz=1920, conf 0.01 — только «tree»);
на кропе, увеличенном ×3, находит «drone» со score ≈0.26. Поэтому кадр режется на тайлы и каждый тайл
увеличивается ×2 (tile_scale). Дорого (≈ 6 тайлов на кадр) — годится для проверки/подтверждения, не для каждого кадра.

Требует: pip install git+https://github.com/ultralytics/CLIP.git (текстовый энкодер для set_classes);
веса yoloe-11s-seg.pt (скачиваются ultralytics, лицензия AGPL-3.0).
"""
from __future__ import annotations

import cv2
import numpy as np

from ..detect_base import Detection, Detector


class OpenVocabDrone(Detector):
    name = "drone_yoloe"

    def __init__(self, weights: str = "yoloe-11s-seg.pt", prompts=("drone", "quadcopter"), conf: float = 0.08,
                 tile: tuple[int, int] = (640, 540), tile_scale: float = 2.0, device: str = "cpu"):
        from ultralytics import YOLOE

        self.model = YOLOE(weights)
        self.prompts = list(prompts)
        self.model.set_classes(self.prompts, self.model.get_text_pe(self.prompts))
        self.conf, self.tile, self.tile_scale, self.device = conf, tile, tile_scale, device

    def config(self) -> dict:
        return {"name": self.name, "prompts": self.prompts, "conf": self.conf, "tile": list(self.tile), "tile_scale": self.tile_scale}

    def detect(self, cam_id, frame_idx, image, camera_model) -> list[Detection]:
        H, W = image.shape[:2]
        tw, th = self.tile
        out = []
        for y0 in range(0, H, th):
            for x0 in range(0, W, tw):
                crop = image[y0:y0 + th, x0:x0 + tw]
                big = cv2.resize(crop, None, fx=self.tile_scale, fy=self.tile_scale, interpolation=cv2.INTER_CUBIC)
                r = self.model.predict(big, imgsz=int(max(big.shape[:2])), conf=self.conf, verbose=False, device=self.device)[0]
                for s, b in zip(r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                    bb = (x0 + b[0] / self.tile_scale, y0 + b[1] / self.tile_scale, x0 + b[2] / self.tile_scale, y0 + b[3] / self.tile_scale)
                    out.append(Detection(bb, ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2), float(s), "drone", self.name))
        out.sort(key=lambda d: -d.score)
        return out[:3]
