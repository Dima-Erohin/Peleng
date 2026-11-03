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
  fill_before_init — "first" | "empty": что писать в кадры до первой оценки;
  road          — (необязательно) описание дороги для viewer.json, см. offline/road.py: {"enabled", "width_m",
                  "width_factor", "h_pivot_m", ...}; выключить: --no-road или --set truck.road.enabled=false.

Переопределение из командной строки: --set truck.tracker.accel_std=2 --set drone.detections=/path/dets.json
(значение разбирается как JSON, иначе строка) или целиком файлом --objects-file objects.json.
"""
from __future__ import annotations

import copy
import json

# --- Геометрия грузовика (свойства модели; тот же грузовик в финальных видео) ---------------------------------
# TRUCK_H_PIVOT_M — высота pivot модели над поверхностью дороги (м). Нужна offline/road.py: дорога рисуется
# на высоте z_pivot − h_pivot. Получена ОДИН раз на тестовом наборе по эталону: на кадрах, где грузовик виден
# >= 2 камерами, берётся луч через нижний-центр рамки детекции (низ рамки = контакт колёс с дорогой), он
# пересекается с вертикалью через эталонный pivot (точка луча, ближайшая к вертикали), и z_pivot − z_луча даёт
# высоту. Рамки — только те, что содержат проекцию pivot. Для камер с большим углом склонения ошибка огромна
# (смещение колеса вдоль луча × tg угла: камеры 2 и 4 дают 3..7 м), поэтому оставлены лучи с углом склонения
# < 5° (камеры 1, 3, 5; 371 луч): медиана 3.12 м, межквартильный 2.93..3.35 м, по третям набора 3.0 / 3.2 / 3.3 м;
# по отдельным камерам 3.01 / 3.23 / 3.04 м. Округлено до 3.1 м, точность ~ ±0.2 м.
TRUCK_H_PIVOT_M = 3.1
# Ширина грузовика (м): типовая ширина полуприцепа/кузова 2.5 м (совпадает с априором габаритов DimsEstimator
# в tracker3d.py); по видео не измерялась. Ширина дороги по умолчанию = 1.6 × это значение = 4.0 м.
TRUCK_WIDTH_M = 2.5
# Дорога в viewer.json (offline/road.py): enabled=false — не писать; width_m — задать ширину явно.
TRUCK_ROAD = {"enabled": True, "h_pivot_m": TRUCK_H_PIVOT_M, "truck_width_m": TRUCK_WIDTH_M, "width_factor": 1.6,
              "width_m": None}

PRESETS: dict[str, dict] = {
    # Грузовик (эталон — Loc объекта = pivot модели, проецируется у переда кабины).
    "truck": {
        "name": "truck",
        "out_csv": "truck.csv",
        # детектор грузовика из offline/detectors/truck.py (фон -> кроп -> YOLO11-seg); дешёвая альтернатива —
        # "yolo+bg" с detector_kw {"yolo_weights": "yolo11n.pt", "yolo_imgsz": 640, "yolo_classes": [...]}
        "detector": "offline.detectors.truck:TruckDetector",
        "detector_kw": {},
        "detections": None,
        # Центр рамки -> EKF; выход — pivot модели грузовика = центр + 10.9 м вперёд по курсу и −0.45 м по высоте.
        # Это свойство модели грузовика (тот же грузовик в финальных видео), измерено один раз по эталону
        # тестового набора: медиана (эталон − оценка центра) в осях курса по кадрам с >= 2 камерами, устойчиво
        # по третям набора (11.1 / 9.9 / 11.0 м). σ ускорения 5 м/с² (по вертикали 0.5 — наземная машина),
        # σ пикселя 2 + 0.1·размер рамки — выбраны по устойчивости (leave-one-camera-out), см. README.
        # Создание трека по одной камере выключено: без второй камеры дальность ненадёжна (хуже при выпадении камер).
        "tracker": {"model": "cv", "accel_std": 5.0, "accel_std_z": 0.5, "point": "center", "meas": "point",
                    "anchor": "center", "out_offset_fwd_m": 10.9, "out_offset_up_m": -0.45,
                    "size_range": True, "px_sigma_rel": 0.10, "size_rel_min": 0.04, "size_rel_growth": 0.01,
                    "mono_init": False},
        # если основной детектор в камере ничего не нашёл — сильная YOLO в окне вокруг причинного прогноза;
        # если трека нет — раз в 10 кадров на весь кадр (1280), чтобы начать/возобновить трек без вычитания фона
        "roi": {"weights": "yolo11m.pt", "imgsz": 640, "conf": 0.15, "classes": ["truck", "car", "bus"],
                "only_if_empty": True, "full_every": 10, "full_imgsz": 1280},
        "truth": True,
        "fill_before_init": "first",
        "road": dict(TRUCK_ROAD),
    },
    # Грузовик детектором позы (offline/detectors/truck_pose.py, YOLO-pose, обученная на этом грузовике):
    # keypoint pivot = эталонная точка -> триангулируется сама точка pivot, смещения не нужны.
    # Веса: detector_kw.weights или переменная окружения TRUCK_WEIGHTS. Вычитание фона не используется —
    # подходит для движущихся камер.
    "truck_pose": {
        "name": "truck",
        "out_csv": "truck.csv",
        "detector": "offline.detectors.truck_pose:TruckPoseDetector",
        "detector_kw": {},
        "detections": None,
        "tracker": {"model": "cv", "accel_std": 5.0, "accel_std_z": 0.5, "point": "kp:pivot", "meas": "point",
                    "anchor": "center", "size_range": True, "px_sigma_rel": 0.10, "size_rel_min": 0.04,
                    "size_rel_growth": 0.01, "mono_init": False},
        "roi": None,
        "truth": True,
        "fill_before_init": "first",
        "road": dict(TRUCK_ROAD),
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
