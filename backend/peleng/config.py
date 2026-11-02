"""Загрузка конфигурации: backend/config.json (если есть) поверх значений по умолчанию."""
from __future__ import annotations

import copy
import json
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent
DATA_DIR = REPO_DIR / "data"

DEFAULTS: dict = {
    "server": {"host": "0.0.0.0", "port": 8443, "https": True},
    # Камеры, которые сервер открывает сам. Телефоны через браузер (/cam) добавляются автоматически.
    # source: 0 — встроенная вебкамера; "http://192.168.1.23:8080/video" — приложение IP Webcam / DroidCam
    "cameras": [],
    "default_hfov_deg": 68.0,
    "board": {
        # A4: 5×4, клетка 0.05, маркер 0.037. A3: клетка 0.07, маркер 0.052.
        # ВАЖНО: после печати измерьте клетку линейкой и впишите реальный размер.
        "squares_x": 5,
        "squares_y": 4,
        "square_m": 0.05,
        "marker_m": 0.037,
        "dictionary": "DICT_4X4_50",
        "height_m": 0.0,
    },
    "pose": {"model": "yolo11n-pose.pt", "imgsz": 640, "conf": 0.4, "device": None},
    "pipeline": {
        "max_fps": 15,
        "max_skew_s": 0.15,
        "kp_conf": 0.35,
        "epi_thresh_px": 25.0,
        "reproj_thresh_px": 15.0,
        "mono_fallback": True,
        # уточнять фокус по кадрам доски при калибровке положения (если оптика не калибровалась). Грубо, см. calibration.py
        "refine_focal_on_extrinsics": False,
    },
    "tracking": {"gate_m": 1.0, "max_age_s": 1.5, "min_hits": 2},
    # Размер комнаты для отрисовки коробки [ширина X, глубина Y, высота Z] и смещение центра доски от центра комнаты
    "room": {"size_m": None, "board_offset_m": [0.0, 0.0]},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict:
    if path is None:
        p = BACKEND_DIR / "config.json"
        path = p if p.exists() else None
    if path is None:
        return copy.deepcopy(DEFAULTS)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return _merge(DEFAULTS, data)
