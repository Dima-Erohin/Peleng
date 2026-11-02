"""Главный цикл обработки: кадры -> 2D-скелеты -> 3D -> трекинг -> сообщение для фронтенда."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable

from .pose import Person2D
from .reconstruct import reconstruct
from .sources import CameraRegistry, Frame
from .tracking import Tracker

Detector = Callable[[list[tuple[str, Frame]]], list[list[Person2D]]]


class Pipeline:
    def __init__(self, cfg: dict, registry: CameraRegistry, detector: Detector, publish: Callable[[dict], None], sessions_dir: Path):
        self.cfg = cfg
        self.registry = registry
        self.detector = detector
        self.publish = publish
        self.sessions_dir = Path(sessions_dir)
        tcfg = cfg.get("tracking", {})
        self.tracker = Tracker(gate_m=tcfg.get("gate_m", 1.0), max_age_s=tcfg.get("max_age_s", 1.5), min_hits=tcfg.get("min_hits", 2))
        self.last_dets: dict[str, tuple[float, list[Person2D]]] = {}
        self._last_seq: dict[str, int] = {}
        self._seq = 0
        self._running = False
        self._rec = None
        self._rec_path: Path | None = None
        self._rec_lock = threading.Lock()
        self.stats: dict = {}
        self.paused = False
        self.hello_provider: Callable[[], dict] | None = None

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
        for cid, f in batch:
            self._last_seq[cid] = f.seq

        t0 = time.time()
        results = self.detector(batch)
        t_det = time.time() - t0
        dets: dict[str, list[Person2D]] = {}
        for (cid, f), persons in zip(batch, results):
            dets[cid] = persons
            self.last_dets[cid] = (f.ts, persons)

        tref = max(f.ts for _, f in batch)
        cams = {}
        used_dets = {}
        for cid, f in batch:
            h, w = f.image.shape[:2]
            m = self.registry.model(cid, (w, h))
            if m is not None and m.calibrated and tref - f.ts <= max_skew:
                cams[cid] = m
                used_dets[cid] = dets[cid]
        skew = (tref - min(f.ts for c, f in batch if c in cams)) if cams else 0.0
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
        }
        msg = {
            "type": "frame",
            "t": round(tref, 4),
            "seq": self._seq,
            "persons": persons,
            "detections": {cid: [p.to_dict() for p in ps] for cid, ps in dets.items()},
            "stats": self.stats,
        }
        with self._rec_lock:
            if self._rec:
                self._rec.write(json.dumps(msg, ensure_ascii=False) + "\n")
        return msg

    def _loop(self) -> None:
        max_fps = self.cfg.get("pipeline", {}).get("max_fps", 15)
        last_fps_t = time.time()
        n = 0
        while self._running:
            t0 = time.time()
            msg = None
            if not self.paused:
                try:
                    msg = self.step()
                except Exception as e:  # noqa: BLE001 — цикл не должен падать
                    print(f"[pipeline] ошибка: {e!r}")
                    time.sleep(0.5)
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
