"""Детектор грузовика YOLO-pose: рамка + keypoint pivot (эталонная точка). Без вычитания фона — камеры могут двигаться.

Веса (best.pt из train_truck_pose.py) не лежат в репозитории: weights=... или переменная окружения TRUCK_WEIGHTS.
Возвращает Detection(cls="truck", bbox=рамка, point_px=pivot (keypoint 0), score=conf рамки,
extra={"pivot":(x,y), "kp_conf":..., "box_center":(x,y)}). Причинно: каждый кадр независим.
"""
from __future__ import annotations

import os

import numpy as np

from ..detect_base import Detection, Detector

# 3D-модель грузовика в системе (вдоль курса x, влево y, вверх z) относительно pivot; порядок — как в truck_dataset
# (idx=1+si*4+ti*2+zi). Размеры — те же, что в разметке (truck_dataset.FRONT/REAR/HALF_W/HEIGHT/Z_BOTTOM).
_FRONT, _REAR, _HW, _H, _ZB = 0.3, 22.0, 1.4, 4.5, -1.7
MODEL_PTS = np.array([[0, 0, 0]] + [[s, t, z] for s in (_FRONT, -_REAR) for t in (_HW, -_HW) for z in (_ZB, _ZB + _H)], np.float64)


def pnp_pivot(kp: np.ndarray, camera_model, min_conf: float = 0.3):
    """Поза по одной камере: kp (9,3) [x,y,conf] в пикселях -> (pivot в мировой системе (3,), курс-вектор (3,), ошибка репроекции px) или None.

    Нужны ≥6 точек с conf ≥ min_conf (включая углы). Метрика/ошибки зависят от точности 3D-габаритов (они оценочные)."""
    import cv2

    ok = kp[:, 2] >= min_conf
    if ok.sum() < 6:
        return None
    obj, img = MODEL_PTS[ok], kp[ok, :2].astype(np.float64)
    K = camera_model.K
    good, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, camera_model.dist, flags=cv2.SOLVEPNP_EPNP, reprojectionError=12.0,
                                               iterationsCount=200)
    if not good:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, camera_model.dist, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, camera_model.dist)
    err = float(np.median(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)))
    Rm, _ = cv2.Rodrigues(rvec)
    Xc = tvec.reshape(3)  # pivot в системе камеры
    Xw = camera_model.R.T @ (Xc - camera_model.t)
    fwd_w = camera_model.R.T @ (Rm @ np.array([1.0, 0, 0]))
    return Xw, fwd_w, err


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
        self.imgsz = imgsz or int(os.environ.get("TRUCK_IMGSZ") or self.model.overrides.get("imgsz", 800))

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
            allkp = kp[i].tolist() if kp is not None and len(kp) > i else []  # [[x,y,conf]]*9: pivot + 8 углов
            out.append(Detection(tuple(b), (px, py), float(s), "truck", self.name,
                                 {"pivot": (px, py), "kp_conf": kc, "box_center": c, "kp": allkp}))
        return out
