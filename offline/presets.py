"""Настройки объектов по умолчанию (пресеты). Один прогон run.py обрабатывает несколько объектов —
у каждого свой детектор, свой трекер и свой выходной CSV.

Поля описания объекта:
  name          — имя (папка результатов <out>/<name>/);
  out_csv       — имя выходного CSV в формате эталона (кладётся в <out>/);
  detector      — "yolo+bg" | "yolo" | "bg" | "модуль:Класс" (например offline.detectors.drone:DroneDetector);
  detector_kw   — аргументы конструктора детектора;
  detections    — путь к готовому JSON детекций (если задан — детектор не запускается);
  tracker       — параметры TrackerConfig (см. tracker3d.py);
  roi           — сильная модель в окне вокруг прогноза трекера: {"weights", "imgsz", "conf", "classes"} или null;
  truth         — сравнивать с эталоном из папки данных (если он есть);
  fill_before_init — "first" | "empty": что писать в кадры до первой оценки.

Переопределение из командной строки: --set truck.tracker.accel_std=2 --set drone.detections=/path/dets.json
(значение разбирается как JSON, иначе строка) или целиком файлом --objects-file objects.json.
"""
from __future__ import annotations

import copy
import json

PRESETS: dict[str, dict] = {
    # Грузовик (эталон тестового набора — ПЕРЁД кабины). Центр объекта — триангуляция центров рамок + EKF;
    # выходная точка — перёд по курсу: центр + L/2 · направление скорости, длина L оценивается причинно по
    # рамкам (DimsEstimator). Вертикальное ускорение наземной машины мало.
    "truck": {
        "name": "truck",
        "out_csv": "truck.csv",
        # детектор грузовика из offline/detectors/truck.py (фон -> кроп -> YOLO11-seg); дешёвая альтернатива —
        # "yolo+bg" с detector_kw {"yolo_weights": "yolo11n.pt", "yolo_imgsz": 640, "yolo_classes": [...]}
        "detector": "offline.detectors.truck:TruckDetector",
        "detector_kw": {},
        "detections": None,
        # центр рамки -> EKF; выход — «перёд по курсу» (центр + L/2·направление скорости, L — DimsEstimator).
        # σ ускорения 7 м/с² по горизонтали — компенсирует манёвры и «дрожание» центра рамки (подобрано на
        # тестовом наборе: 3 -> 11.6 м RMSE, 5 -> 9.8, 7 -> 8.7, 10 -> 9.2); по вертикали — 0.5 (наземная машина).
        "tracker": {"model": "cv", "accel_std": 7.0, "accel_std_z": 0.5, "point": "center", "meas": "point",
                    "anchor": "front", "size_range": True, "px_sigma_rel": 0.15},
        "roi": None,
        "truth": True,
        "fill_before_init": "first",
    },
    # Дрон: универсальная 3D-модель (без «земли»), допускает зависание. Детектор — из offline/detectors/
    # (делается отдельно); точка — та, что даёт детектор. Априорный размер (м) даёт дальность по одной камере.
    "drone": {
        "name": "drone",
        "out_csv": "drone.csv",
        "detector": "offline.detectors.drone:DroneTemplateDetector",
        "detector_kw": {},
        "detections": None,
        # на тестовых видео дрон виден только камерой 5 (все камеры неподвижны по положению -> параллакса нет):
        # глубина — только по видимой ширине рамки при допущении о размере (object_size_m, по ширине);
        # такие кадры помечаются mode=mono_size_prior. σ ускорения мала — дрон висит/движется плавно.
        "tracker": {"model": "cv", "accel_std": 1.0, "accel_std_z": None, "point": "detector", "meas": "point",
                    "anchor": "center", "size_range": True, "object_size_m": 0.5, "size_dim": "width",
                    "init_min_score": 0.3},
        "roi": None,
        "truth": False,
        "fill_before_init": "first",
    },
    # Любой объект детектором yolo+bg без допущений (центр рамки).
    "generic": {
        "name": "object",
        "out_csv": "trajectory.csv",
        "detector": "yolo+bg",
        "detector_kw": {},
        "detections": None,
        "tracker": {"model": "cv", "accel_std": 3.0, "point": "center", "meas": "point", "anchor": "center"},
        "roi": None,
        "truth": True,
        "fill_before_init": "first",
    },
}


def get_specs(names: list[str], objects_file: str | None = None, sets: list[str] | None = None) -> list[dict]:
    if objects_file:
        specs = json.load(open(objects_file, encoding="utf-8"))
        specs = specs if isinstance(specs, list) else [specs]
        base = []
        for s in specs:
            b = copy.deepcopy(PRESETS.get(s.get("preset", s.get("name")), PRESETS["generic"]))
            _deep_update(b, s)
            base.append(b)
        specs = base
    else:
        specs = []
        for n in names:
            if n not in PRESETS:
                raise ValueError(f"нет пресета {n}; есть: {', '.join(PRESETS)}")
            specs.append(copy.deepcopy(PRESETS[n]))
    by_name = {s["name"]: s for s in specs}
    for item in sets or []:
        key, _, val = item.partition("=")
        parts = key.split(".")
        if parts[0] not in by_name:
            raise ValueError(f"--set {item}: нет объекта {parts[0]}")
        try:
            v = json.loads(val)
        except json.JSONDecodeError:
            v = val
        d = by_name[parts[0]]
        for p in parts[1:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return specs


def _deep_update(a: dict, b: dict) -> None:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_update(a[k], v)
        else:
            a[k] = v
