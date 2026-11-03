"""Расширение «Кадры сессии»: при записи сессии сохраняет уменьшенные кадры камер, при --replay показывает их в превью.

Запись: хуки старта/остановки записи конвейера + frame_hook (кладёт кадры в очередь потока-писателя, см. peleng/frames.py).
Replay: камеры из hello записи регистрируются в реестре как ReplaySource, кадр которых — записанный кадр, ближайший по времени
к текущему моменту проигрывания (state["replay_t"]). Поэтому штатные /api/cameras/{id}/snapshot.jpg | mjpeg и /api/state
работают без изменений, а у пирамидок камер в 3D появляются текстуры.

Настройки — cfg["recording"]["frames"] (значения по умолчанию — peleng/frames.py: DEFAULTS); "enabled": false отключает запись кадров.
Старые записи (без папки кадров) проигрываются как раньше.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import cv2
import numpy as np

from .. import frames as fr
from ..geometry import CameraModel
from ..sources import Frame, Source


class ReplaySource(Source):
    kind = "replay"

    def __init__(self, cam_id: str, name: str | None, index: fr.FrameIndex, state: dict):
        super().__init__(cam_id, name)
        self.index = index
        self.state = state
        self._cache: tuple[int, Frame] | None = None

    def latest(self) -> Frame | None:
        i = self.index.nearest(self.id, self.state.get("replay_t"))
        if i is None:
            return None
        with self._lock:
            if self._cache and self._cache[0] == i:
                return self._cache[1]
        img = cv2.imdecode(np.fromfile(str(self.index.path(self.id, i)), np.uint8), cv2.IMREAD_COLOR) if self.index.path(self.id, i).is_file() else None
        if img is None:
            return None
        f = Frame(img, self.index.times[self.id][i], i + 1)
        with self._lock:
            self._cache = (i, f)
        return f

    @property
    def online(self) -> bool:
        return bool(self.index.times.get(self.id))

    @property
    def fps(self) -> float:
        return self.index.fps(self.id)


def _model_from_hello(c: dict) -> CameraModel | None:
    try:
        if "K" not in c or "size" not in c:
            return None
        d = {"K": c["K"], "dist": [0.0] * 5, "size": c["size"], "meta": {"intrinsics": c.get("intrinsics", "replay")}}
        if c.get("calibrated") and "R" in c and "t" in c:
            d["R"], d["t"] = c["R"], c["t"]
        return CameraModel.from_dict(d)
    except Exception:  # noqa: BLE001
        return None


def _register_replay(ctx) -> None:
    path = ctx.state.get("replay_path")
    if not path:
        return
    index = fr.FrameIndex(fr.frames_dir_for(path))
    if not index.cams:
        return  # старая запись без кадров — всё как раньше
    hello_cams: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            first = json.loads(f.readline() or "{}")
        if first.get("type") == "hello":
            hello_cams = first.get("cameras") or []
    except (OSError, ValueError):
        pass
    # калибровка настоящих камер лежит в data/calib/<id>.json; кнопки «Сброс»/«Оптика» во вьювере при replay
    # не должны её трогать — направляем записи калибровки в несуществующую папку (ошибка вместо порчи данных)
    ctx.registry.calib_dir = Path(ctx.data_dir) / "calib_replay_unused"
    by_id = {c["id"]: c for c in hello_cams if "id" in c}
    for cam in index.cams:
        c = by_id.get(cam, {})
        m = _model_from_hello(c)
        if m is not None:
            ctx.registry.set_model(cam, m, save=False)  # до add(): иначе подтянется калибровка из data/calib
        src = ReplaySource(cam, c.get("name"), index, ctx.state)
        src.info = c.get("info") or {}
        ctx.registry.add(src)
    print(f"[frames] replay: кадры камер {', '.join(index.cams)} из {index.root}")


def register(app, ctx) -> None:
    if ctx.state.get("mode") == "replay":
        _register_replay(ctx)
        return
    s = fr.settings(ctx.cfg)
    if not s["enabled"]:
        return
    box: dict = {"w": None}
    lock = threading.Lock()

    def on_start(path) -> None:
        with lock:
            box["w"] = fr.FrameWriter(fr.frames_dir_for(path), s)

    def on_stop(path) -> None:
        with lock:
            w, box["w"] = box["w"], None
        if w is not None:
            w.close()
            print(f"[frames] сохранено кадров: {w.saved}, пропущено при перегрузке: {w.dropped}")

    def frame_hook(ctx_f) -> None:
        w = box["w"]
        if w is None:
            return
        for cid, f in ctx_f.batch:
            w.offer(cid, f.ts, f.image)

    ctx.pipeline.record_start_hooks.append(on_start)
    ctx.pipeline.record_stop_hooks.append(on_stop)
    ctx.pipeline.frame_hooks.append(frame_hook)
