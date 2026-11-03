"""Детектор грузовика: движение (предложения областей) + YOLO-seg на кропах + точка «перёд кабины».

Зачем так (подробности и цифры — в README.md):
* COCO-детекторы на полном кадре 1920x1080 находят грузовик плохо, когда он далёкий/перекрыт деревьями
  (камеры 2-4: 40-60% кадров даже у yolo11x/RT-DETR). Вычитание фона находит движущееся пятно почти всегда,
  но его рамка неточная (часть кузова/тени).
* Поэтому: причинное вычитание фона (BackgroundDetector, с компенсацией поворота камеры) даёт области-кандидаты,
  вокруг каждой вырезается кроп (масштаб ×2-3 относительно полного кадра), на кропе работает YOLO11-seg
  (классы truck/car/bus) — рамка и МАСКА грузовика.
* Эталонная точка (из CSV) — ПЕРЁД кабины, а не центр рамки (центр рамки отстоит на ~0.4·ширину рамки).
  Оценка: самая «передняя» точка маски в направлении движения в кадре v (v — по предыдущим кадрам, причинно)
  минус небольшая поправка kappa·w вдоль v. Если скорость неизвестна — центр рамки.

Причинность: для кадра k используются только кадры <= k (фон, трек скорости копятся из прошлых вызовов).
"""
from __future__ import annotations

import cv2
import numpy as np

from ..basic_detectors import BackgroundDetector
from ..detect_base import Detection, Detector

TRUCK_CLASSES = ("truck", "car", "bus")


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def front_point(poly: np.ndarray | None, box, v: np.ndarray | None, kappa: float = 0.034, top_frac: float = 0.03):
    """Точка «перёд» по маске/рамке и направлению движения v (пиксели/кадр, вектор в кадре)."""
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    if v is None or np.linalg.norm(v) < 1e-6:
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    u = v / np.linalg.norm(v)
    if poly is not None and len(poly) >= 3:
        s = poly @ u
        top = poly[s >= s.max() - top_frac * max(s.max() - s.min(), 1e-6)]
        q = top.mean(0)
    else:  # без маски: ближайший к направлению угол/край рамки (по ходу), высота ~60% от верха
        q = np.array([x2 if u[0] > 0 else x1, y1 + 0.6 * (y2 - y1)])
    q = q - kappa * w * u
    return (float(q[0]), float(q[1]))


class _Track:
    """Лёгкий причинный трек одного грузовика на камеру: положение рамки и скорость в кадре."""

    def __init__(self):
        self.box = None
        self.k = -10**9
        self.v = None  # np.array(2) px/кадр
        self.c = None

    def warp(self, H):
        """Перенос состояния трека в систему координат нового кадра (компенсация поворота камеры)."""
        if self.c is None or H is None:
            return
        c2 = H @ np.array([self.c[0], self.c[1], 1.0])
        c2 = c2[:2] / c2[2]
        d = c2 - self.c
        self.c = c2
        if self.box is not None:
            self.box = [self.box[0] + d[0], self.box[1] + d[1], self.box[2] + d[0], self.box[3] + d[1]]

    def predict_center(self, k):
        if self.c is None:
            return None
        dt = k - self.k
        return self.c + (self.v * dt if self.v is not None else 0)

    def update(self, k, box, alpha=0.5):
        c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        if self.c is not None and 0 < k - self.k <= 10:
            vn = (c - self.c) / (k - self.k)
            if self.v is None:
                self.v = vn
            elif np.linalg.norm(vn - self.v) <= 4.0 + 0.5 * np.linalg.norm(self.v):  # отсечь скачки рамки
                self.v = alpha * vn + (1 - alpha) * self.v
        elif self.c is None or k - self.k > 10:
            self.v = None
        self.c, self.box, self.k = c, box, k


class TruckDetector(Detector):
    name = "truck"

    def __init__(self, seg_weights: str = "yolo11m-seg.pt", conf: float = 0.2, crop_imgsz: int = 640,
                 full_weights: str | None = None, full_every: int = 0, full_imgsz: int = 1280,
                 min_blob_area: float = 400.0, max_rois: int = 3, kappa: float = 0.034,
                 min_speed: float = 0.3, device: str = "cpu"):
        from ultralytics import YOLO

        self.seg_weights = seg_weights
        self.seg = YOLO(seg_weights)
        self.full = YOLO(full_weights) if full_weights else None
        self.full_every = full_every
        self.full_imgsz = full_imgsz
        self.conf = conf
        self.crop_imgsz = crop_imgsz
        self.min_blob_area = min_blob_area
        self.max_rois = max_rois
        self.kappa = kappa
        self.min_speed = min_speed
        self.device = device
        self.bg = BackgroundDetector()
        self.tracks: dict[int, _Track] = {}
        self.prev_cam: dict[int, object] = {}
        names = self.seg.names
        self.ids = [i for i, n in names.items() if n in TRUCK_CLASSES]

    def config(self) -> dict:
        return {"name": self.name, "seg": self.seg_weights, "full": getattr(self.full, "ckpt_path", None) and str(self.full_every),
                "conf": self.conf, "crop_imgsz": self.crop_imgsz, "min_blob_area": self.min_blob_area,
                "max_rois": self.max_rois, "kappa": self.kappa, "full_every": self.full_every}

    # ---- вспомогательное -------------------------------------------------
    def _rois(self, blobs, track, k, shape):
        H, W = shape[:2]
        rois = []
        for d in sorted(blobs, key=lambda d: -d.extra.get("area", 0))[: self.max_rois + 2]:
            if d.extra.get("area", 0) < self.min_blob_area:
                continue
            rois.append(list(d.bbox))
        pc = track.predict_center(k) if track is not None else None
        if pc is not None and k - track.k <= 15:
            b = track.box
            w, h = b[2] - b[0], b[3] - b[1]
            rois.append([pc[0] - w / 2, pc[1] - h / 2, pc[0] + w / 2, pc[1] + h / 2])
        # слить пересекающиеся
        merged = []
        for r in rois:
            for m in merged:
                if _iou(r, m) > 0.2 or (r[0] < m[2] and r[2] > m[0] and r[1] < m[3] and r[3] > m[1]):
                    m[0], m[1], m[2], m[3] = min(m[0], r[0]), min(m[1], r[1]), max(m[2], r[2]), max(m[3], r[3])
                    break
            else:
                merged.append(list(r))
        out = []
        for r in merged[: self.max_rois]:
            cx, cy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
            side = max(1.7 * max(r[2] - r[0], r[3] - r[1]), 320.0)
            side = min(side, min(H, W))
            x0 = int(np.clip(cx - side / 2, 0, W - side))
            y0 = int(np.clip(cy - side / 2, 0, H - side))
            out.append((x0, y0, int(side)))
        return out

    def _run_seg(self, image, x0, y0, side):
        crop = image[y0:y0 + side, x0:x0 + side]
        r = self.seg.predict(crop, imgsz=self.crop_imgsz, conf=self.conf, classes=self.ids, verbose=False, device=self.device)[0]
        res = []
        if r.boxes is None:
            return res
        s = side / float(crop.shape[1]) if crop.shape[1] else 1.0
        for i, (c, sc, b) in enumerate(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())):
            box = [b[0] * s + x0, b[1] * s + y0, b[2] * s + x0, b[3] * s + y0]
            # детекция, обрезанная границей кропа, ненадёжна — понижаем score
            edge = b[0] < 2 or b[1] < 2 or b[2] > crop.shape[1] - 2 or b[3] > crop.shape[0] - 2
            poly = None
            if r.masks is not None and i < len(r.masks.xy) and len(r.masks.xy[i]) >= 3:
                poly = r.masks.xy[i] * s + np.array([x0, y0])
            res.append((box, float(sc) * (0.7 if edge else 1.0), r.names[int(c)], poly))
        return res

    # ---- интерфейс --------------------------------------------------------
    def detect(self, cam_id, frame_idx, image, camera_model) -> list[Detection]:
        k = frame_idx
        blobs = self.bg.detect(cam_id, k, image, camera_model)  # вызывается для КАЖДОГО кадра: копит фон
        tr = self.tracks.setdefault(cam_id, _Track())
        pcam = self.prev_cam.get(cam_id)
        if pcam is not None and camera_model is not None and np.abs(pcam.R - camera_model.R).max() > 1e-9 \
                and np.linalg.norm(pcam.center - camera_model.center) < 1e-3:
            K = camera_model.K
            tr.warp(K @ camera_model.R @ pcam.R.T @ np.linalg.inv(K))  # H: кадр k-1 -> кадр k
        self.prev_cam[cam_id] = camera_model
        cands = []
        for (x0, y0, side) in self._rois(blobs, tr, k, image.shape):
            cands += self._run_seg(image, x0, y0, side)
        if self.full is not None and self.full_every and (k % self.full_every == 0 or not cands):
            r = self.full.predict(image, imgsz=self.full_imgsz, conf=self.conf, verbose=False, device=self.device,
                                  classes=[i for i, n in self.full.names.items() if n in TRUCK_CLASSES])[0]
            for c, sc, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                cands.append((list(b), float(sc), r.names[int(c)], None))
        # NMS между кандидатами разных кропов
        cands.sort(key=lambda c: -c[1])
        keep = []
        for c in cands:
            if all(_iou(c[0], q[0]) < 0.5 for q in keep):
                keep.append(c)
        # выбрать «главный» (для трека): лучший score рядом с предсказанием
        pc = tr.predict_center(k)
        if keep and pc is not None and k - tr.k <= 15:
            def dist(c):
                return np.hypot((c[0][0] + c[0][2]) / 2 - pc[0], (c[0][1] + c[0][3]) / 2 - pc[1])
            near = [c for c in keep if dist(c) < 0.8 * (c[0][2] - c[0][0]) + 60]
            main = max(near, key=lambda c: c[1]) if near else keep[0]
        else:
            main = keep[0] if keep else None
        if main is not None:
            tr.update(k, main[0])
        out = []
        for c in keep:
            box, sc, cls, poly = c
            v = tr.v if (c is main and tr.v is not None and np.linalg.norm(tr.v) >= self.min_speed) else None
            if c is not main and tr.v is not None and np.linalg.norm(tr.v) >= self.min_speed:
                v = tr.v  # других кандидатов тоже считаем движущимися в ту же сторону (одинаковое направление — оценка)
            pt = front_point(poly, box, v, self.kappa)
            out.append(Detection(tuple(box), pt, sc, "truck", "truck",
                                 {"center": ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
                                  "v": None if v is None else [float(v[0]), float(v[1])],
                                  "main": c is main, "coco_cls": cls}))
        return out
