"""Подключение общей геометрии из backend/peleng (CameraModel, triangulate_robust, ...)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_BACKEND = ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
