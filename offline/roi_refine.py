"""Уточняющий детектор в окне (ROI) вокруг прогноза трекера — причинно.

Зачем: лёгкая YOLO на целом кадре (640 px по длинной стороне) плохо видит грузовик (кадр уменьшается в 3 раза,
вид сзади/сверху, частичное заслонение). Сильная модель (yolo11m/l/x) на всём кадре в полном разрешении — секунды
на кадр. Поэтому сильная модель запускается только на вырезке вокруг места, где трекер ожидает объект на кадре k
(прогноз по кадрам < k), в исходном разрешении (вырезка ≥ imgsz px — масштаб не хуже 1:1 для небольших объектов).
Если трек потерян — окна ставятся вокруг лучших кандидатов базового детектора (по кадру k).

Кэш: результаты по ключу (камера, кадр, окно, квантованное до сетки) хранятся в JSON — повторные прогоны трекинга
с близкими параметрами почти не запускают нейросеть.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .detect_base import Detection


class RoiRefiner:
    def __init__(self, weights: str = "yolo11m.pt", imgsz: int = 640, classes=("truck", "car", "bus"), conf: float = 0.1,
                 min_side: int = 640, ctx: float = 1.6, grid: int = 32, cache_path: Path | None = None, device: str = "cpu"):
        self.weights = weights
        self.imgsz = imgsz
        self.classes = tuple(classes) if classes else ()
        self.conf = conf
        self.min_side = min_side
        self.ctx = ctx
        self.grid = grid
        self.device = device
        self.model = None
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, list] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text())
            except Exception:
                self.cache = {}
        self.n_run = 0
        self.n_cached = 0
        self.t_model = 0.0
        self._dirty = 0

    def config(self) -> dict:
        return {"weights": self.weights, "imgsz": self.imgsz, "classes": list(self.classes), "conf": self.conf,
                "min_side": self.min_side, "ctx": self.ctx}

    def _load(self):
        if self.model is None:
            from ultralytics import YOLO

            self.model = YOLO(self.weights)
            names = self.model.names
            self.class_ids = [i for i, n in names.items() if n in self.classes] if self.classes else None

    def window(self, bbox, size) -> tuple[int, int, int] | None:
        """Квадратное окно вокруг ожидаемой рамки: сторона = max(min_side, ctx * размер), квантовано по сетке."""
        W, H = size
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(self.min_side, self.ctx * max(x2 - x1, y2 - y1))
        side = int(min(side, H if side <= H else max(W, H)))
        # квантование размера (шаг ×1.25) и положения (сетка) — для кэша
        q = self.min_side
        while q < side:
            q = int(q * 1.25)
        side = min(q, max(W, H))
        g = self.grid
        x0 = int(round((cx - side / 2) / g) * g)
        y0 = int(round((cy - side / 2) / g) * g)
        x0 = int(np.clip(x0, 0, max(0, W - side)))
        y0 = int(np.clip(y0, 0, max(0, H - side)))
        return x0, y0, side

    def detect(self, cam_id: int, frame: int, image: np.ndarray, win: tuple[int, int, int],
               imgsz: int | None = None) -> list[Detection]:
        x0, y0, side = win
        imgsz = imgsz or self.imgsz
        key = f"{cam_id}:{frame}:{x0}:{y0}:{side}" + (f":{imgsz}" if imgsz != self.imgsz else "")
        if key in self.cache:
            self.n_cached += 1
            return [Detection.from_list(a) for a in self.cache[key]]
        self._load()
        H, W = image.shape[:2]
        crop = image[y0:min(H, y0 + side), x0:min(W, x0 + side)]
        t = time.time()
        r = self.model.predict(crop, imgsz=imgsz, conf=self.conf, classes=self.class_ids, verbose=False,
                               device=self.device)[0]
        self.t_model += time.time() - t
        self.n_run += 1
        out = []
        ch, cw = crop.shape[:2]
        if r.boxes is not None:
            for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                # рамка упирается в край окна, который не является краем кадра — объект обрезан окном: не берём
                cut = (b[0] < 2 and x0 > 0) or (b[1] < 2 and y0 > 0) or (b[2] > cw - 2 and x0 + cw < W) or \
                      (b[3] > ch - 2 and y0 + ch < H)
                if cut:
                    continue
                bb = (b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0)
                out.append(Detection(bb, ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2), float(s), r.names[int(c)], "roi"))
        self.cache[key] = [d.to_list() for d in out]
        self._dirty += 1
        if self._dirty >= 200:
            self.save()
        return out

    def detect_many(self, items: list) -> list[list[Detection]]:
        """Несколько окон одним вызовом сети (онлайн-режим): items — [(image, win)], без кэша."""
        if not items:
            return []
        self._load()
        crops, metas = [], []
        for image, (x0, y0, side) in items:
            H, W = image.shape[:2]
            crops.append(image[y0:min(H, y0 + side), x0:min(W, x0 + side)])
            metas.append((x0, y0, W, H))
        t = time.time()
        rs = self.model.predict(crops, imgsz=self.imgsz, conf=self.conf, classes=self.class_ids, verbose=False,
                                device=self.device)
        self.t_model += time.time() - t
        self.n_run += len(items)
        out = []
        for r, crop, (x0, y0, W, H) in zip(rs, crops, metas):
            ch, cw = crop.shape[:2]
            dets = []
            if r.boxes is not None:
                for c, sc, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                    cut = (b[0] < 2 and x0 > 0) or (b[1] < 2 and y0 > 0) or (b[2] > cw - 2 and x0 + cw < W) or                           (b[3] > ch - 2 and y0 + ch < H)
                    if cut:
                        continue
                    bb = (b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0)
                    dets.append(Detection(bb, ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2), float(sc), r.names[int(c)], "roi"))
            out.append(dets)
        return out

    def save(self):
        if self.cache_path and self._dirty:
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.cache))
            tmp.replace(self.cache_path)
            self._dirty = 0


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(u, 1e-6)


def merge(base: list[Detection], roi: list[Detection], iou_thr: float = 0.3) -> list[Detection]:
    """Детекции окна главнее: базовые, пересекающиеся с ними, отбрасываются; дубликаты окон — тоже."""
    out: list[Detection] = []
    for d in sorted(roi, key=lambda d: -d.score):
        if all(_iou(d.bbox, o.bbox) < 0.5 for o in out):
            out.append(d)
    for d in base:
        if all(_iou(d.bbox, o.bbox) < iou_thr for o in out):
            out.append(d)
    return out
