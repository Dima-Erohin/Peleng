"""Главный цикл обработки: кадры -> 2D-скелеты -> 3D -> трекинг -> сообщение для фронтенда.

Телефоны снимают кадры по команде: как только пачка кадров взята в обработку, всем телефонам сразу уходит
«capture» на следующую. Пока YOLO считает, новые кадры уже едут по сети. Кадры с разных телефонов
сняты почти одновременно, и трафика ровно столько, сколько успеваем обработать.
"""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .geometry import CameraModel
from .pose import Person2D
from .reconstruct import reconstruct
from .sources import CameraRegistry, Frame
from .tracking import Tracker

Detector = Callable[[list[tuple[str, Frame]]], list[list[Person2D]]]


@dataclass
class FrameContext:
    """Что видит расширение (peleng/ext) на каждом обработанном кадре."""

    batch: list[tuple[str, Frame]]  # (камера, кадр); кадр: .image (BGR), .ts
    cams: dict[str, CameraModel]  # откалиброванные камеры, прошедшие проверку синхронности (под размер кадра)
    dets: dict[str, list[Person2D]]  # 2D-скелеты по камерам
    t: float  # время пачки
    msg: dict  # сообщение "frame" — расширение может добавлять свои поля


class Pipeline:
    def __init__(self, cfg: dict, registry: CameraRegistry, detector: Detector, publish: Callable[[dict], None], sessions_dir: Path):
        self.cfg = cfg
        self.registry = registry
        self.detector = detector
        self.publish = publish
        self.sessions_dir = Path(sessions_dir)
        tcfg = cfg.get("tracking", {})
        self.tracker = Tracker(gate_m=tcfg.get("gate_m", 1.0), max_age_s=tcfg.get("max_age_s", 1.5), min_hits=tcfg.get("min_hits", 2))
        self.last_dets: dict[str, tuple[float, list[Person2D], int]] = {}  # камера -> (время, люди, ширина кадра)
        self._last_seq: dict[str, int] = {}
        self._seq = 0
        self._running = False
        self._rec = None
        self._rec_path: Path | None = None
        self._rec_lock = threading.Lock()
        self.stats: dict = {}
        self.paused = False
        self.hello_provider: Callable[[], dict] | None = None
        self.frame_hooks: list[Callable[[FrameContext], None]] = []  # расширения: дополнить сообщение "frame"
        self.reset_hooks: list[Callable[[], None]] = []  # расширения: сбросить свои треки
        self._full_res: dict[str, int] = {}  # камера (или "*") -> сколько потребителей ждут полное разрешение
        self._full_lock = threading.Lock()
        self._pending: tuple[float, dict[str, int]] | None = None  # (время запроса, {камера: seq на момент запроса})
        # сглаженные длительности: детекция и «команда -> кадр на сервере». По ним запрос отправляется так,
        # чтобы кадры приехали к концу детекции: и синхронно, и без лишней задержки
        self._det_ema = 0.0
        self._cap_ema = 0.05
        self._cap_measured = True

    # --- кадры по команде ---
    @contextmanager
    def full_res(self, cam_ids: list[str] | None = None):
        """Пока открыт контекст, телефоны шлют кадры в полном разрешении (калибровка, поиск доски)."""
        keys = list(cam_ids) if cam_ids else ["*"]
        with self._full_lock:
            for k in keys:
                self._full_res[k] = self._full_res.get(k, 0) + 1
        try:
            yield
        finally:
            with self._full_lock:
                for k in keys:
                    self._full_res[k] -= 1
                    if self._full_res[k] <= 0:
                        del self._full_res[k]

    def _wants_full(self, cam_id: str) -> bool:
        return bool(self._full_res.get("*") or self._full_res.get(cam_id))

    def request_frames(self) -> None:
        width = int(self.cfg.get("pipeline", {}).get("track_width", 640))
        asked = {}
        for cid, src in list(self.registry.sources.items()):
            if src.request(0 if self._wants_full(cid) else width):
                f = src.latest()
                asked[cid] = f.seq if f else 0
        self._pending = (time.time(), asked)
        self._cap_measured = False

    def _capture_timeout(self) -> float:
        return float(self.cfg.get("pipeline", {}).get("capture_timeout_s", 0.3))

    def _waiting(self) -> bool:
        """Запрошенные кадры ещё едут: не все телефоны ответили и таймаут не вышел."""
        if self._pending is None:
            return False
        t_req, asked = self._pending
        if time.time() - t_req > self._capture_timeout():
            return False
        t_last = t_req
        for cid, seq in asked.items():
            src = self.registry.sources.get(cid)
            f = src.latest() if src else None
            if src is not None and src.on_demand and (f is None or f.seq == seq):
                return True
            if f is not None:
                t_last = max(t_last, f.t_recv)
        if asked and not self._cap_measured:
            self._cap_measured = True
            self._cap_ema = 0.8 * self._cap_ema + 0.2 * (t_last - t_req)
        return False

    def _schedule_request(self) -> None:
        """Запросить следующие кадры так, чтобы они приехали к концу текущей детекции."""
        delay = self._det_ema - self._cap_ema - 0.01
        if delay <= 0.005:
            self.request_frames()
        else:
            threading.Timer(delay, self.request_frames).start()

    def _request_stale(self, interval: float) -> bool:
        return self._pending is None or time.time() - self._pending[0] >= interval

    # --- запись сессии ---
    def start_recording(self) -> str:
        with self._rec_lock:
            if self._rec:
                return str(self._rec_path)
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
            self._rec_path = self.sessions_dir / time.strftime("session_%Y%m%d_%H%M%S.jsonl")
            self._rec = open(self._rec_path, "w", encoding="utf-8")
            if self.hello_provider:
                self._rec.write(json.dumps(self.hello_provider(), ensure_ascii=False) + "\n")
            return str(self._rec_path)

    def stop_recording(self) -> str | None:
        with self._rec_lock:
            if not self._rec:
                return None
            self._rec.close()
            self._rec = None
            return str(self._rec_path)

    @property
    def recording(self) -> str | None:
        return str(self._rec_path) if self._rec else None

    # --- обработка ---
    def step(self) -> dict | None:
        pcfg = self.cfg.get("pipeline", {})
        max_skew = pcfg.get("max_skew_s", 0.15)
        now = time.time()
        batch: list[tuple[str, Frame]] = []
        fresh = False
        for cid, src in list(self.registry.sources.items()):
            f = src.latest()
            if f is None or now - f.ts > 1.0:
                continue
            if self._last_seq.get(cid) != f.seq:
                fresh = True
            batch.append((cid, f))
        if not batch or not fresh:
            return None
        # камеры-потоки (вебкамера, IP Webcam) — берём кадр, ближайший по времени к снятым по команде
        srcs = self.registry.sources
        t_cmd = [f.ts for cid, f in batch if srcs[cid].on_demand]
        if t_cmd:
            t_target = max(t_cmd)
            batch = [(cid, f if srcs[cid].on_demand else (srcs[cid].nearest(t_target) or f)) for cid, f in batch]
        for cid, f in batch:
            self._last_seq[cid] = f.seq
        self._schedule_request()  # следующие кадры снимаются, пока считаем эти

        t0 = time.time()
        results = self.detector(batch)
        t_det = time.time() - t0
        self._det_ema = 0.8 * self._det_ema + 0.2 * t_det if self._det_ema else t_det
        dets: dict[str, list[Person2D]] = {}
        for (cid, f), persons in zip(batch, results):
            dets[cid] = persons
            self.last_dets[cid] = (f.ts, persons, f.image.shape[1] if f.image is not None else 1)

        tref = max(f.ts for _, f in batch)
        cams = {}
        used_dets = {}
        for cid, f in batch:
            h, w = f.image.shape[:2]
            m = self.registry.model(cid, (w, h))
            if m is not None and m.calibrated and tref - f.ts <= max_skew:
                cams[cid] = m
                used_dets[cid] = dets[cid]
        skew = tref - min(f.ts for _, f in batch)  # разброс времени съёмки по всем камерам пачки
        obs = reconstruct(
            cams,
            used_dets,
            kp_conf=pcfg.get("kp_conf", 0.35),
            epi_thresh_px=pcfg.get("epi_thresh_px", 25.0),
            reproj_thresh_px=pcfg.get("reproj_thresh_px", 15.0),
            mono_fallback=pcfg.get("mono_fallback", True),
        )
        persons = self.tracker.update(tref, obs)
        self._seq += 1
        proc = time.time() - t0
        self.stats = {
            "proc_ms": round(proc * 1000, 1),
            "detect_ms": round(t_det * 1000, 1),
            "latency_ms": round((time.time() - tref) * 1000, 1),
            "skew_ms": round(skew * 1000, 1),
            "cams_used": sorted(cams),
            "fps": self.stats.get("fps"),
        }
        msg = {
            "type": "frame",
            "t": round(tref, 4),
            "seq": self._seq,
            "persons": persons,
            "detections": {cid: [p.to_dict() for p in ps] for cid, ps in dets.items()},
            # размер кадра, в пикселях которого detections (у телефонов при трекинге — уменьшенный, track_width)
            "frame_size": {cid: [int(f.image.shape[1]), int(f.image.shape[0])] for cid, f in batch if f.image is not None},
            "stats": self.stats,
        }
        if self.frame_hooks:
            ctx = FrameContext(batch, cams, dets, tref, msg)
            for hook in list(self.frame_hooks):
                try:
                    hook(ctx)
                except Exception as e:  # noqa: BLE001 — расширение не должно ронять конвейер
                    print(f"[pipeline] ошибка в {getattr(hook, '__name__', hook)}: {e!r}")
            self.stats["proc_ms"] = round((time.time() - t0) * 1000, 1)
        with self._rec_lock:
            if self._rec:
                self._rec.write(json.dumps(msg, ensure_ascii=False) + "\n")
        return msg

    def reset_tracking(self) -> None:
        """Сбросить треки (например, после перекалибровки) — и у конвейера, и у расширений."""
        self.tracker.reset()
        for fn in list(self.reset_hooks):
            fn()

    def _loop(self) -> None:
        max_fps = self.cfg.get("pipeline", {}).get("max_fps", 15)
        last_fps_t = time.time()
        n = 0
        while self._running:
            t0 = time.time()
            msg = None
            if self.paused:
                # обработка стоит, но превью и калибровке кадры всё равно нужны
                if self._request_stale(1.0 / max_fps):
                    self.request_frames()
            elif self._waiting():
                time.sleep(0.003)
                continue
            else:
                try:
                    msg = self.step()
                except Exception as e:  # noqa: BLE001 — цикл не должен падать
                    print(f"[pipeline] ошибка: {e!r}")
                    time.sleep(0.5)
                # нечего обрабатывать (телефон только подключился или потерял команду) — спросить снова
                if msg is None and self._request_stale(self._capture_timeout()):
                    self.request_frames()
            if msg is not None:
                n += 1
                if time.time() - last_fps_t >= 1.0:
                    self.stats["fps"] = round(n / (time.time() - last_fps_t), 1)
                    n, last_fps_t = 0, time.time()
                msg["stats"]["fps"] = self.stats.get("fps")
                self.publish(msg)
                time.sleep(max(0.0, 1.0 / max_fps - (time.time() - t0)))
            else:
                time.sleep(0.005)

    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name="pipeline").start()

    def stop(self) -> None:
        self._running = False
        self.stop_recording()
