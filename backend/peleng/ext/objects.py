"""Расширение «Объекты»: отслеживание предметов (мяч, машинка, дрон, метки ArUco) — поле "objects" в сообщении frame.

Настройки — config.json -> "objects" (см. DEFAULTS), режим и классы меняются через REST без перезапуска:
  GET  /api/objects/         — режим, классы, состояние детектора, число объектов
  POST /api/objects/         — {"mode": "off|yolo|aruco", "classes": [...], "tag_size_m": 0.05}
  POST /api/objects/reset    — сбросить треки объектов
"""
from __future__ import annotations

import contextlib
import threading
import time

import numpy as np
from fastapi import HTTPException

from ..objects import (
    COCO_CLASSES, TAG_DICT, ArucoObjectDetector, ObjectTracker, class_filter, demo_objects, reconstruct_objects, synth_detections,
)

MODES = ("off", "yolo", "aruco")
DEFAULTS = {
    "mode": "off",  # по умолчанию выключено — не тратить CPU
    "classes": [],  # пусто — все классы COCO, кроме person
    "model": "yolo11n.pt",
    "imgsz": 640,
    "conf": 0.35,
    "device": None,
    "tag_dictionary": TAG_DICT,
    "tag_size_m": 0.05,  # сторона чёрного квадрата метки (для mono по одной камере)
    "full_res": False,  # aruco: просить у телефонов кадры в полном разрешении (мелкие метки видно дальше)
    "epi_thresh_px": 25.0,
    "reproj_thresh_px": 15.0,
    "mono_fallback": True,
    "gate_m": 1.0,
    "max_age_s": 1.0,
    "min_hits": 2,
    "accel_std": 6.0,
}


def register(app, ctx) -> None:
    st = {**DEFAULTS, **ctx.cfg.get("objects", {})}
    if st["mode"] not in MODES:
        st["mode"] = "off"
    demo = ctx.state.get("mode") == "demo"
    tracker = ObjectTracker(st["gate_m"], st["max_age_s"], st["min_hits"], st["accel_std"])
    aruco = ArucoObjectDetector(st["tag_dictionary"])
    yolo: dict = {"det": None, "status": "не загружена", "loading": False}
    info: dict = {"count": 0, "ms": 0.0}
    lock = threading.Lock()
    full_res = contextlib.ExitStack()
    rng = np.random.default_rng(0)

    def load_yolo():
        try:
            from ..objects import YoloObjectDetector

            print(f"[objects] загрузка модели {st['model']} …")
            det = YoloObjectDetector(st["model"], st["imgsz"], st["conf"], st["device"])
            yolo["det"] = det
            yolo["status"] = f"{st['model']} ({det.device})"
            print(f"[objects] готово: {yolo['status']}")
        except Exception as e:  # noqa: BLE001
            yolo["status"] = f"ошибка: {e}"
            print(f"[objects] YOLO недоступен ({e})")
        finally:
            yolo["loading"] = False

    def apply_mode():
        """Подготовить выбранный режим: модель в фоне, полное разрешение для меток."""
        full_res.close()
        if st["mode"] == "yolo" and yolo["det"] is None and not yolo["loading"] and not demo:
            yolo["loading"] = True
            yolo["status"] = "загрузка…"
            threading.Thread(target=load_yolo, daemon=True, name="objects-yolo").start()
        if st["mode"] == "aruco" and st["full_res"] and not demo:
            full_res.enter_context(ctx.pipeline.full_res())

    def detect(fctx) -> dict[str, list]:
        mode = st["mode"]
        if demo:  # в кадрах демо объектов нет — проецируем синтетические
            keep = class_filter(st["classes"])
            items = [it for it in demo_objects(fctx.t) if (it[0].startswith("tag:") if mode == "aruco" else keep(it[0]) and not it[0].startswith("tag:"))]
            return {cid: synth_detections(m, items, rng, 0.7) for cid, m in fctx.cams.items()}
        out: dict[str, list] = {}
        if mode == "aruco":
            for cid, f in fctx.batch:
                if f.image is not None:
                    out[cid] = aruco(f.image)
        elif mode == "yolo" and yolo["det"] is not None:
            frames = [(cid, f.image) for cid, f in fctx.batch if f.image is not None]
            res = yolo["det"]([im for _, im in frames], class_filter(st["classes"]))
            out = {cid: ds for (cid, _), ds in zip(frames, res)}
        return out

    def hook(fctx) -> None:
        if st["mode"] == "off":
            fctx.msg["objects"] = []
            return
        t0 = time.time()
        with lock:
            dets = detect(fctx)
            used = {c: d for c, d in dets.items() if c in fctx.cams}
            obs = reconstruct_objects(fctx.cams, used, st["epi_thresh_px"], st["reproj_thresh_px"], st["mono_fallback"], st["tag_size_m"])
            objects = tracker.update(fctx.t, obs)
        fctx.msg["objects"] = objects
        fctx.msg["object_detections"] = {c: [d.to_dict() for d in ds] for c, ds in dets.items()}
        info["count"] = len(objects)
        info["ms"] = round((time.time() - t0) * 1000, 1)
        fctx.msg["stats"]["objects_ms"] = info["ms"]

    def reset() -> None:
        with lock:
            tracker.reset()

    ctx.pipeline.frame_hooks.append(hook)
    ctx.pipeline.reset_hooks.append(reset)
    apply_mode()

    def state() -> dict:
        return {
            "mode": st["mode"],
            "modes": list(MODES),
            "classes": list(st["classes"]),
            "all_classes": COCO_CLASSES,
            "detector": "demo" if demo else (yolo["status"] if st["mode"] == "yolo" else ("ArUco " + st["tag_dictionary"] if st["mode"] == "aruco" else "выкл")),
            "tag_dictionary": st["tag_dictionary"],
            "tag_size_m": st["tag_size_m"],
            "count": info["count"] if st["mode"] != "off" else 0,
            "proc_ms": info["ms"],
        }

    @app.get("/api/objects")
    @app.get("/api/objects/")
    async def api_objects():
        return state()

    @app.post("/api/objects")
    @app.post("/api/objects/")
    async def api_objects_set(body: dict):
        mode = body.get("mode", st["mode"])
        if mode not in MODES:
            raise HTTPException(400, f"режим должен быть одним из {MODES}")
        if "classes" in body:
            cl = body["classes"]
            if isinstance(cl, str):
                cl = cl.split(",")
            st["classes"] = [c.strip() for c in cl if c and c.strip()]
        if body.get("tag_size_m"):
            st["tag_size_m"] = float(body["tag_size_m"])
        if mode != st["mode"]:
            st["mode"] = mode
            reset()
            info["count"] = 0
            apply_mode()
        return state()

    @app.post("/api/objects/reset")
    async def api_objects_reset():
        reset()
        return {"ok": True}
