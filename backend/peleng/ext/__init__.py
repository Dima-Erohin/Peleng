"""Расширения сервера: каждый модуль peleng/ext/<имя>.py подключается автоматически.

Модуль объявляет функцию

    def register(app: FastAPI, ctx: ExtContext) -> None

и в ней добавляет свои REST-эндпоинты (лучше с префиксом /api/<имя>/), хуки конвейера
(ctx.pipeline.frame_hooks — дополнить сообщение "frame", ctx.pipeline.reset_hooks — сбросить свои треки)
и т. п. Свои настройки расширение читает из ctx.cfg.get("<имя>", {}) со значениями по умолчанию у себя.

Интерфейс во вьювере — файлы backend/web/ext/*.js, они тоже подключаются автоматически (см. index.html).
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExtContext:
    cfg: dict
    registry: Any  # sources.CameraRegistry
    pipeline: Any  # pipeline.Pipeline
    hub: Any  # server.Hub — hub.publish(dict) рассылает сообщение всем вьюверам
    board: Any  # calibration.Board
    state: dict  # общее состояние сервера (mode: live | demo | replay, …)
    data_dir: Path


def load_extensions(app, ctx: ExtContext) -> list[str]:
    """Порядок: по ORDER модуля (по умолчанию 50), затем по имени. Хуки конвейера вызываются в том же порядке,
    так что расширение, которое дополняет чужие поля (например, точность для "objects"), ставит ORDER побольше."""
    mods = []
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        try:
            mods.append((info.name, importlib.import_module(f"{__name__}.{info.name}")))
        except Exception as e:  # noqa: BLE001 — сломанное расширение не должно ронять сервер
            print(f"[ext] {info.name}: не загрузилось ({e!r})")
    loaded = []
    for name, mod in sorted(mods, key=lambda nm: (getattr(nm[1], "ORDER", 50), nm[0])):
        if not hasattr(mod, "register"):
            continue
        try:
            mod.register(app, ctx)
            loaded.append(name)
        except Exception as e:  # noqa: BLE001
            print(f"[ext] {name}: ошибка при подключении ({e!r})")
    return loaded
