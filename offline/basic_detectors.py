"""Базовые детекторы для проверки конвейера на тестовом наборе (грузовик).

* BackgroundDetector («bg») — причинное вычитание фона: фон = экспоненциальное среднее ПРОШЛЫХ кадров
  (кадр k сравнивается с фоном, построенным по кадрам < k, и только потом добавляется в фон).
  Для поворачивающейся камеры фон переносится в текущий кадр гомографией чистого вращения
  H = K · R_k · R_{k-1}^T · K^-1 (центр камеры неподвижен; поворот известен из CSV). Если центр камеры
  сдвинулся — фон сбрасывается и копится заново. Области, только что «въехавшие» в кадр, не дают
  срабатываний, пока не накопят warmup кадров.
* YoloDetector («yolo») — ultralytics YOLO (по умолчанию yolo11n.pt, классы truck/car/bus), кадр
  уменьшается до imgsz по длинной стороне.
* CombinedDetector («yolo+bg») — объединение: пятна фона, лежащие внутри рамки YOLO, отбрасываются,
  остальные добавляются с пониженным score.
"""
from __future__ import annotations

import importlib
import inspect
import os

import cv2
import numpy as np

from .detect_base import Detection, Detector


class BackgroundDetector(Detector):
    """Причинное вычитание фона (см. описание модуля).

    Фон — экспоненциальное среднее прошлых кадров. Опции (по умолчанию выключены — на тестовом наборе они дали
    больше ложных пятен): selective — выборочное обновление (передний план фон не портит, нет «следа»; пиксель,
    бывший передним планом дольше absorb_frames кадров, вливается в фон); shadow — тени (темнее фона при том же
    цвете) не считаются передним планом."""

    name = "bg"
    version = 2

    def __init__(self, scale: int = 2, alpha: float = 0.03, thresh: int = 25, warmup: int = 10,
                 min_area_px: float = 80.0, max_blobs: int = 8, blur: int = 3, selective: bool = False,
                 absorb_frames: int = 150, shadow: bool = False, shadow_ratio: tuple = (0.4, 0.92),
                 shadow_chroma: float = 0.06):
        self.scale = scale
        self.alpha = alpha
        self.thresh = thresh
        self.warmup = warmup
        self.min_area_px = min_area_px  # в пикселях полного кадра
        self.max_blobs = max_blobs
        self.blur = blur
        self.selective = selective
        self.absorb_frames = absorb_frames
        self.shadow = shadow
        self.shadow_ratio = shadow_ratio
        self.shadow_chroma = shadow_chroma
        self.state: dict[int, dict] = {}

    def config(self) -> dict:
        return {"name": self.name, "v": self.version, "scale": self.scale, "alpha": self.alpha, "thresh": self.thresh,
                "warmup": self.warmup, "min_area_px": self.min_area_px, "max_blobs": self.max_blobs,
                "selective": self.selective, "absorb": self.absorb_frames, "shadow": self.shadow}

    def _small(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        im = cv2.resize(image, (w // self.scale, h // self.scale), interpolation=cv2.INTER_AREA)
        if self.blur > 1:
            im = cv2.GaussianBlur(im, (self.blur, self.blur), 0)
        return im.astype(np.float32)

    def _Ks(self, cam) -> np.ndarray:
        K = cam.K.copy()
        K[:2] /= self.scale
        return K

    def detect(self, cam_id, frame_idx, image, camera_model) -> list[Detection]:
        img = self._small(image)
        st = self.state.get(cam_id)
        if st is not None and camera_model is not None and st["cam"] is not None:
            prev = st["cam"]
            if np.linalg.norm(prev.center - camera_model.center) > 1e-3:
                st = None  # камера переместилась — фон заново
            elif np.abs(prev.R - camera_model.R).max() > 1e-9:
                # чистый поворот: перенос фона гомографией H = K R_k R_{k-1}^T K^-1
                Ks = self._Ks(camera_model)
                Hm = Ks @ camera_model.R @ prev.R.T @ np.linalg.inv(Ks)
                dsize = (img.shape[1], img.shape[0])
                st["bg"] = cv2.warpPerspective(st["bg"], Hm, dsize, flags=cv2.INTER_LINEAR)
                st["cnt"] = cv2.warpPerspective(st["cnt"], Hm, dsize, flags=cv2.INTER_NEAREST, borderValue=0)
                st["fgc"] = cv2.warpPerspective(st["fgc"], Hm, dsize, flags=cv2.INTER_NEAREST, borderValue=0)
        if st is None:
            st = {"bg": img.copy(), "cnt": np.ones(img.shape[:2], np.float32), "fgc": np.zeros(img.shape[:2], np.float32),
                  "cam": camera_model}
            self.state[cam_id] = st
            return []
        dets: list[Detection] = []
        valid = st["cnt"] >= self.warmup
        bg = st["bg"]
        diff = np.abs(img - bg).max(axis=2)
        fg = (diff > self.thresh) & valid
        if self.shadow:
            ib, ii = bg.sum(axis=2) + 1e-3, img.sum(axis=2) + 1e-3
            ratio = ii / ib
            chroma = np.abs(img / ii[..., None] - bg / ib[..., None]).max(axis=2)
            shadow = (ratio > self.shadow_ratio[0]) & (ratio < self.shadow_ratio[1]) & (chroma < self.shadow_chroma)
            fg &= ~shadow
        m = fg.astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        if valid.any():
            n, _, stats, cen = cv2.connectedComponentsWithStats(m)
            s = self.scale
            cand = []
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if area * s * s < self.min_area_px:
                    continue
                cand.append((area, x, y, w, h, cen[i]))
            cand.sort(key=lambda c: -c[0])
            for area, x, y, w, h, c in cand[: self.max_blobs]:
                bbox = (x * s, y * s, (x + w) * s, (y + h) * s)
                score = float(min(1.0, 0.2 + 0.1 * np.log10(max(area * s * s, 1.0))))
                dets.append(Detection(bbox, ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2), score, "motion", "bg",
                                      {"area": int(area * s * s)}))
        # обновление фона — ПОСЛЕ детекции (кадр k не участвует в своём же фоне)
        a = self.alpha
        fresh = st["cnt"] < 1
        fgd = cv2.dilate(m, np.ones((5, 5), np.uint8)).astype(bool)
        st["fgc"] = np.where(fgd, st["fgc"] + 1, 0).astype(np.float32)
        absorb = (st["fgc"] > self.absorb_frames) if self.selective else np.zeros_like(fgd)
        upd = ((~fgd) | absorb) if self.selective else np.ones_like(fgd)
        newbg = np.where(upd[..., None], (1 - a) * bg + a * img, bg)
        newbg = np.where((fresh | absorb)[..., None], img, newbg)
        st["fgc"][absorb] = 0
        st["bg"] = newbg.astype(np.float32)
        st["cnt"] = np.minimum(st["cnt"] + 1, 10000).astype(np.float32)
        st["cam"] = camera_model
        return dets


class YoloDetector(Detector):
    name = "yolo"

    def __init__(self, weights: str = "yolo11n.pt", classes=("truck", "car", "bus"), conf: float = 0.15,
                 imgsz: int = 640, device: str = "cpu"):
        from ultralytics import YOLO

        self.weights = weights
        self.model = YOLO(weights)
        names = self.model.names
        self.class_names = tuple(classes) if classes else ()
        self.class_ids = [i for i, n in names.items() if n in self.class_names] if classes else None
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def config(self) -> dict:
        return {"name": self.name, "weights": str(self.weights), "classes": list(self.class_names),
                "conf": self.conf, "imgsz": self.imgsz}

    def detect(self, cam_id, frame_idx, image, camera_model) -> list[Detection]:
        r = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, classes=self.class_ids,
                               verbose=False, device=self.device)[0]
        out = []
        if r.boxes is None:
            return out
        for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
            out.append(Detection(tuple(b), ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), float(s), r.names[int(c)], "yolo"))
        return out


def _ioa(inner, outer) -> float:
    """Доля площади inner внутри outer."""
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / a


def combine_dets(dy: list[Detection], db: list[Detection], bg_score_scale: float = 0.6) -> list[Detection]:
    """Объединение YOLO и вычитания фона: пятна внутри рамок YOLO (или содержащие их) отбрасываются."""
    out = list(dy)
    for d in db:
        if any(_ioa(d.bbox, y.bbox) > 0.5 or _ioa(y.bbox, d.bbox) > 0.5 for y in dy):
            continue
        out.append(Detection(d.bbox, d.point_px, d.score * bg_score_scale, d.cls, d.source, d.extra))
    return out


class CombinedDetector(Detector):
    name = "yolo+bg"

    def __init__(self, yolo: YoloDetector | None = None, bg: BackgroundDetector | None = None, bg_score_scale: float = 0.6):
        self.yolo = yolo or YoloDetector()
        self.bg = bg or BackgroundDetector()
        self.bg_score_scale = bg_score_scale

    def config(self) -> dict:
        return {"name": self.name, "yolo": self.yolo.config(), "bg": self.bg.config(), "bg_score_scale": self.bg_score_scale}

    def detect(self, cam_id, frame_idx, image, camera_model) -> list[Detection]:
        dy = self.yolo.detect(cam_id, frame_idx, image, camera_model)
        db = self.bg.detect(cam_id, frame_idx, image, camera_model)
        out = list(dy)
        for d in db:
            if any(_ioa(d.bbox, y.bbox) > 0.5 or _ioa(y.bbox, d.bbox) > 0.5 for y in dy):
                continue
            d.score *= self.bg_score_scale
            out.append(d)
        return out


def make_detector(name: str, **kw) -> Detector:
    """«bg», «yolo», «yolo+bg» или «модуль:Класс» (внешний детектор с тем же интерфейсом)."""
    if name == "bg":
        return BackgroundDetector(**kw)
    if name == "yolo":
        return YoloDetector(**kw)
    if name in ("yolo+bg", "bg+yolo"):
        ykw = {k[5:]: v for k, v in kw.items() if k.startswith("yolo_")}
        bkw = {k[3:]: v for k, v in kw.items() if k.startswith("bg_")}
        return CombinedDetector(YoloDetector(**ykw), BackgroundDetector(**bkw))
    if ":" in name:
        mod, cls = name.split(":", 1)
        klass = getattr(importlib.import_module(mod), cls)
        dev = os.environ.get("TRUCK_DEVICE")
        if dev and "device" not in kw and "device" in inspect.signature(klass).parameters:
            kw = {**kw, "device": dev}  # GPU-хост: все сети прогона на видеокарте (TRUCK_DEVICE=0)
        return klass(**kw)
    raise ValueError(f"неизвестный детектор {name}")
