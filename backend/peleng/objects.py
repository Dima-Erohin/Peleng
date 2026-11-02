"""Отслеживание объектов (не людей): детекция в кадрах -> сопоставление между камерами -> 3D -> трекинг.

Режимы детекции:
  * yolo  — обычная YOLO-детекция (yolo11n.pt, классы COCO), точка объекта — центр рамки;
  * aruco — ArUco-метки (словарь DICT_5X5_100: DICT_4X4_50 занят доской), точка — центр метки.

Сопоставление: метки — по ID; YOLO — тот же класс + эпиполярное расстояние центров, венгерский алгоритм
для каждой пары камер. ≥ 2 камер -> триангуляция ("tri"); одна камера -> "mono":
для YOLO — точка на полу по низу рамки (объект стоит на полу), для метки — solvePnP по известному размеру метки.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import CameraModel, essential_between, ray_plane_z, symmetric_epipolar_px, triangulate_robust
from .reconstruct import Observation3D
from .tracking import Tracker

TAG_DICT = "DICT_5X5_100"

# 80 классов COCO (yolo11n.pt) — чтобы интерфейс мог показать список, пока модель не загружена
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


@dataclass
class Det2D:
    cls: str  # класс COCO ("sports ball") или "tag:<id>" для метки
    xy: np.ndarray  # (2,) точка объекта в пикселях: центр рамки / центр метки
    bbox: np.ndarray  # (4,) x1,y1,x2,y2
    score: float
    corners: np.ndarray | None = None  # (4,2) углы метки (только aruco)

    def to_dict(self) -> dict:
        return {"cls": self.cls, "bbox": [round(float(v), 1) for v in self.bbox], "score": round(float(self.score), 3)}


@dataclass
class ObjectObservation(Observation3D):
    cls: str = ""


def class_filter(classes: list[str] | None):
    """Фильтр классов: пустой список — все, кроме person."""
    wanted = {c.strip() for c in (classes or []) if c and c.strip()}
    if wanted:
        return lambda c: c in wanted
    return lambda c: c != "person"


# ---------------------------------------------------------------------------
# Детекторы
# ---------------------------------------------------------------------------

def _diag_center(c: np.ndarray) -> np.ndarray:
    """Центр четырёхугольника как пересечение диагоналей — проекция центра метки без искажения перспективой."""
    p1, p3 = c[0], c[2]
    p2, p4 = c[1], c[3]
    l1 = np.cross([*p1, 1.0], [*p3, 1.0])
    l2 = np.cross([*p2, 1.0], [*p4, 1.0])
    x = np.cross(l1, l2)
    if abs(x[2]) < 1e-9:
        return c.mean(axis=0)
    return x[:2] / x[2]


class ArucoObjectDetector:
    def __init__(self, dictionary: str = TAG_DICT):
        self.dict_name = dictionary
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(d, p)

    def __call__(self, image: np.ndarray) -> list[Det2D]:
        if image is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids, _ = self.detector.detectMarkers(gray)
        out: dict[int, Det2D] = {}
        if ids is None:
            return []
        for c, i in zip(corners, ids.reshape(-1)):
            c = c.reshape(4, 2).astype(np.float64)
            area = abs(cv2.contourArea(c.astype(np.float32)))
            d = Det2D(f"tag:{int(i)}", _diag_center(c), np.array([*c.min(axis=0), *c.max(axis=0)]), area, c)
            # одна и та же метка дважды в кадре — берём крупнее
            if int(i) not in out or out[int(i)].score < area:
                out[int(i)] = d
        for d in out.values():
            d.score = 1.0
        return list(out.values())


class YoloObjectDetector:
    def __init__(self, model: str = "yolo11n.pt", imgsz: int = 640, conf: float = 0.35, device: str | None = None):
        from ultralytics import YOLO  # тяжёлый импорт — только когда нужен

        from .pose import _auto_device

        self.model = YOLO(model)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device or _auto_device()
        self.names = dict(self.model.names)

    def __call__(self, images: list[np.ndarray], keep=None) -> list[list[Det2D]]:
        keep = keep or class_filter(None)
        out: list[list[Det2D]] = [[] for _ in images]
        idx = [i for i, im in enumerate(images) if im is not None]
        if not idx:
            return out
        results = self.model.predict([images[i] for i in idx], imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
        for i, r in zip(idx, results):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes = r.boxes.xyxy.cpu().numpy().astype(np.float64)
            scores = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for b, s, c in zip(boxes, scores, cls):
                name = self.names.get(int(c), str(c))
                if keep(name):
                    out[i].append(Det2D(name, np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]), b, float(s)))
        return out


# ---------------------------------------------------------------------------
# Сопоставление между камерами
# ---------------------------------------------------------------------------

def associate_objects(cams: dict[str, CameraModel], dets: dict[str, list[Det2D]], thresh_px: float = 25.0) -> list[dict[str, int]]:
    """Группы детекций одного объекта: {камера: индекс детекции}. В группе не больше одной детекции с камеры.

    Метки ("tag:N") — по ID. Остальное — для каждой пары камер венгерский алгоритм по эпиполярному
    расстоянию центров (разные классы несовместимы), затем слияние пар, начиная с самых надёжных.
    """
    ids = [c for c in dets if c in cams and cams[c].calibrated]
    clusters: list[dict[str, int]] = []

    # метки: по ID
    by_tag: dict[str, dict[str, int]] = {}
    for c in ids:
        for i, d in enumerate(dets[c]):
            if d.cls.startswith("tag:"):
                by_tag.setdefault(d.cls, {})[c] = i
    clusters.extend(by_tag.values())

    # YOLO: класс + эпиполярное расстояние
    edges: list[tuple[float, tuple[str, int], tuple[str, int]]] = []
    norm = {c: cams[c].undistort(np.array([d.xy for d in dets[c]])) if dets[c] else np.zeros((0, 2)) for c in ids}
    for ai, a in enumerate(ids):
        ia = [i for i, d in enumerate(dets[a]) if not d.cls.startswith("tag:")]
        if not ia:
            continue
        for b in ids[ai + 1:]:
            ib = [j for j, d in enumerate(dets[b]) if not d.cls.startswith("tag:")]
            if not ib:
                continue
            E = essential_between(cams[a], cams[b])
            C = np.full((len(ia), len(ib)), 1e6)
            for r, i in enumerate(ia):
                for s, j in enumerate(ib):
                    if dets[a][i].cls != dets[b][j].cls:
                        continue
                    C[r, s] = symmetric_epipolar_px(E, norm[a][i:i + 1], norm[b][j:j + 1], cams[a].focal, cams[b].focal)[0]
            for r, s in zip(*linear_sum_assignment(C)):
                if C[r, s] < thresh_px:
                    edges.append((float(C[r, s]), (a, ia[r]), (b, ib[s])))
    edges.sort(key=lambda e: e[0])
    cluster_of: dict[tuple[str, int], int] = {}
    yolo_cl: list[dict[str, int]] = []
    for c in ids:
        for i, d in enumerate(dets[c]):
            if not d.cls.startswith("tag:"):
                cluster_of[(c, i)] = len(yolo_cl)
                yolo_cl.append({c: i})
    for _, u, v in edges:
        cu, cv = cluster_of[u], cluster_of[v]
        if cu == cv:
            continue
        A, B = yolo_cl[cu], yolo_cl[cv]
        if set(A) & set(B):
            continue  # две детекции с одной камеры
        A.update(B)
        for c, i in B.items():
            cluster_of[(c, i)] = cu
        yolo_cl[cv] = {}
    clusters.extend(c for c in yolo_cl if c)
    return clusters


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------

def mono_object(cam_id: str, cam: CameraModel, d: Det2D, tag_size_m: float = 0.05) -> ObjectObservation | None:
    """Одна камера. Метка — solvePnP по известному размеру. YOLO — низ рамки на полу,
    высота центра — по лучу через центр рамки над этой точкой пола."""
    if d.corners is not None:
        if not tag_size_m:
            return None
        s = tag_size_m / 2
        obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, d.corners.reshape(4, 1, 2), cam.K, cam.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok or tvec[2, 0] <= 0:
            return None
        X = cam.R.T @ (tvec.reshape(3) - cam.t.reshape(3))
        return ObjectObservation(X, np.array([X[0], X[1], 0.0]), "mono", [cam_id], dets={cam_id: 0}, cls=d.cls)
    w, h = cam.size
    x1, y1, x2, y2 = d.bbox
    if y2 > h - 3:  # низ рамки обрезан краем кадра — точка на полу неизвестна
        return None
    o, rays = cam.rays(np.array([[(x1 + x2) / 2, y2], d.xy]))
    F = ray_plane_z(o, rays[0], 0.0)
    if F is None or np.linalg.norm(F - o) > 30:
        return None
    dc = rays[1]
    dxy = dc[:2]
    u = float(np.dot(F[:2] - o[:2], dxy) / max(np.dot(dxy, dxy), 1e-9))
    z = float(np.clip(o[2] + u * dc[2], 0.0, 3.0)) if u > 0 else 0.0
    return ObjectObservation(np.array([F[0], F[1], z]), np.array([F[0], F[1], 0.0]), "mono", [cam_id], dets={cam_id: 0}, cls=d.cls)


def reconstruct_objects(
    cams: dict[str, CameraModel],
    dets: dict[str, list[Det2D]],
    epi_thresh_px: float = 25.0,
    reproj_thresh_px: float = 15.0,
    mono_fallback: bool = True,
    tag_size_m: float = 0.05,
) -> list[ObjectObservation]:
    obs: list[ObjectObservation] = []
    singles: list[tuple[str, int]] = []
    for cl in associate_objects(cams, dets, epi_thresh_px):
        if len(cl) >= 2:
            cs = list(cl)
            ds = [dets[c][cl[c]] for c in cs]
            X, err, used = triangulate_robust([cams[c] for c in cs], [d.xy for d in ds], [d.score for d in ds], reproj_thresh_px)
            if X is not None and -1.0 < X[2] < 5.0:
                obs.append(ObjectObservation(X, np.array([X[0], X[1], 0.0]), "tri", sorted(cs[u] for u in used), err, None, dict(cl), cls=ds[0].cls))
                continue
        singles.extend(cl.items())
    if mono_fallback:
        for c, i in singles:
            o = mono_object(c, cams[c], dets[c][i], tag_size_m)
            if o is None:
                continue
            o.dets = {c: i}
            # тот же объект уже найден триангуляцией
            if any(t.cls == o.cls and np.linalg.norm(o.floor[:2] - t.floor[:2]) < 0.5 for t in obs if t.mode == "tri"):
                continue
            obs.append(o)
    return obs


# ---------------------------------------------------------------------------
# Трекинг
# ---------------------------------------------------------------------------

class ObjectTracker:
    """Свой Tracker на каждый класс: мяч не перепутается с чашкой. ID общие с людьми (Track._ids) — не пересекаются."""

    def __init__(self, gate_m: float = 1.0, max_age_s: float = 1.0, min_hits: int = 2, accel_std: float = 6.0,
                 meas_std: dict | None = None):
        self.kw = dict(gate_m=gate_m, max_age_s=max_age_s, min_hits=min_hits, accel_std=accel_std)
        self.meas_std = meas_std or {"tri": (0.02, 0.02), "mono": (0.15, 0.25)}
        self.trackers: dict[str, Tracker] = {}

    def reset(self) -> None:
        self.trackers = {}

    def update(self, t: float, observations: list[ObjectObservation]) -> list[dict]:
        by_cls: dict[str, list[ObjectObservation]] = {}
        for o in observations:
            by_cls.setdefault(o.cls, []).append(o)
        for cls in by_cls:
            if cls not in self.trackers:
                tr = Tracker(**self.kw)
                tr.meas_std = dict(self.meas_std)
                self.trackers[cls] = tr
        out = []
        for cls, tr in list(self.trackers.items()):
            for item in tr.update(t, by_cls.get(cls, [])):
                item.pop("kp3d", None)
                out.append({"id": item["id"], "kind": "object", "cls": cls, **{k: v for k, v in item.items() if k != "id"}})
            if not tr.tracks:
                del self.trackers[cls]
        out.sort(key=lambda o: o["id"])
        return out


# ---------------------------------------------------------------------------
# Демо: синтетические объекты (в демо-режиме кадры не содержат объектов)
# ---------------------------------------------------------------------------

def demo_objects(t: float) -> list[tuple[str, np.ndarray, float]]:
    """[(класс, центр (3,), радиус/половина стороны)] в момент t: мяч прыгает по кругу, коробка с меткой 7 ездит."""
    w = 2 * math.pi / 8
    ball = np.array([1.0 * math.cos(w * t), 1.0 * math.sin(w * t), 0.11 + 0.6 * abs(math.sin(2.3 * t))])
    box = np.array([-0.8 + 0.6 * math.sin(0.5 * t), 0.5 * math.cos(0.5 * t), 0.30])
    cup = np.array([0.5, -0.6, 0.75])
    return [("sports ball", ball, 0.11), ("tag:7", box, 0.05), ("cup", cup, 0.05)]


def synth_detections(cam: CameraModel, items, rng: np.random.Generator | None = None, noise_px: float = 0.0) -> list[Det2D]:
    """Проекция синтетических объектов в камеру: рамка — по сфере радиуса r, метка — горизонтальный квадрат."""
    w, h = cam.size
    out = []
    for cls, X, r in items:
        if cam.depth(X)[0] <= 0.1:
            continue
        nz = (lambda n: rng.normal(0, noise_px, n)) if rng is not None and noise_px > 0 else (lambda n: np.zeros(n))
        c = cam.project(X)[0]
        if not (0 <= c[0] < w and 0 <= c[1] < h):
            continue
        if cls.startswith("tag:"):
            corners3 = X + np.array([[-r, r, 0], [r, r, 0], [r, -r, 0], [-r, -r, 0]])
            corners = cam.project(corners3) + nz(8).reshape(4, 2)
            out.append(Det2D(cls, _diag_center(corners), np.array([*corners.min(axis=0), *corners.max(axis=0)]), 1.0, corners))
        else:
            rr = r * cam.focal / cam.depth(X)[0]
            c = c + nz(2)
            out.append(Det2D(cls, c, np.array([c[0] - rr, c[1] - rr, c[0] + rr, c[1] + rr]), 0.9))
    return out
