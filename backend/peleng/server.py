"""HTTP/WebSocket-сервер.

  GET  /                    — 3D-вьювер + панель управления (для ноутбука)
  GET  /cam                 — страница-камера для телефона (открыть в браузере телефона)
  WS   /ws/push/{cam_id}    — приём кадров с телефона
  WS   /ws/stream           — поток результатов для фронтенда (формат: docs/data-format.md)
  REST /api/...             — калибровка, сканирование комнаты, запись
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .calibration import Board, calibrate_intrinsics, solve_extrinsics
from .config import BACKEND_DIR, DATA_DIR
from .ext import ExtContext, load_extensions
from .geometry import CameraModel
from .pipeline import Pipeline
from .pose import SKELETON
from .room import scan_points
from .sources import CameraRegistry, OpenCVSource, PushSource

WEB_DIR = BACKEND_DIR / "web"
FULL_RES_SWITCH_S = 0.5  # сколько ждать, пока телефоны переключатся на полное разрешение


class Hub:
    """Рассылка сообщений всем подключённым вьюверам (из любого потока)."""

    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queues: set[asyncio.Queue] = set()

    def publish(self, msg: dict) -> None:
        loop = self.loop
        if loop is None:
            return
        text = json.dumps(msg, ensure_ascii=False)
        try:
            loop.call_soon_threadsafe(self._fanout, text)
        except RuntimeError:  # сервер останавливается, цикл событий уже закрыт — конвейер мог не успеть встать
            pass

    def _fanout(self, text: str) -> None:
        for q in list(self.queues):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(text)


def _null_detector(batch):
    return [[] for _ in batch]


def create_app(cfg: dict, demo: bool = False, replay: str | None = None, data_dir: Path = DATA_DIR) -> FastAPI:
    data_dir = Path(data_dir)
    board = Board(cfg["board"])
    registry = CameraRegistry(data_dir / "calib", cfg.get("default_hfov_deg", 68.0))
    hub = Hub()
    state: dict = {"room_points": None, "intr_jobs": {}, "mode": "replay" if replay else ("demo" if demo else "live"),
             "replay_path": replay, "replay_t": None}  # replay_t — время текущего проигрываемого кадра (для ext/frames)

    detector_box: dict = {"fn": _null_detector, "status": "нет"}
    demo_world = None
    if demo:
        from .demo import DemoDetector, DemoSource, DemoWorld

        demo_world = DemoWorld(n_cams=int(cfg.get("demo", {}).get("cameras", 3)), n_people=int(cfg.get("demo", {}).get("people", 2)))
        for cid, m in demo_world.cams.items():
            registry.add(DemoSource(cid, demo_world))
            registry.set_model(cid, m, save=False)
        detector_box["fn"] = DemoDetector(demo_world)
        detector_box["status"] = "demo"
        if cfg.get("room", {}).get("size_m") is None:
            cfg.setdefault("room", {})["size_m"] = list(demo_world.room)
    elif not replay:
        for c in cfg.get("cameras", []):
            registry.add(OpenCVSource(c["id"], c["source"], c.get("name"), c.get("width"), c.get("height")), c.get("hfov_deg"))

    pipeline = Pipeline(cfg, registry, lambda b: detector_box["fn"](b), hub.publish, data_dir / "sessions")

    def hello() -> dict:
        return {
            "type": "hello",
            "mode": state["mode"],
            "server_time": time.time(),
            "cameras": registry.describe(),
            "board": {**board.to_dict(), "corners_world": np.round(board.world_corners(), 4).tolist()},
            "room": cfg.get("room", {}),
            "detector": detector_box["status"],
            "recording": pipeline.recording,
        }

    pipeline.hello_provider = hello

    def load_detector():
        try:
            from .pose import YoloPoseDetector

            p = cfg["pose"]
            print(f"[pose] загрузка модели {p['model']} …")
            det = YoloPoseDetector(p["model"], p.get("imgsz", 640), p.get("conf", 0.4), p.get("device"))
            detector_box["fn"] = lambda batch: det([f.image for _, f in batch])
            detector_box["status"] = f"{p['model']} ({det.device})"
            print(f"[pose] готово: {detector_box['status']}")
        except Exception as e:  # noqa: BLE001
            detector_box["status"] = f"ошибка: {e}"
            print(f"[pose] YOLO недоступен ({e}). Калибровка работает, детекции людей не будет.")

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        # запуск
        hub.loop = asyncio.get_running_loop()
        for src in registry.sources.values():
            src.start()
        if replay:
            threading.Thread(target=_replay_loop, args=(replay,), daemon=True).start()
        else:
            if not demo:
                threading.Thread(target=load_detector, daemon=True).start()
            pipeline.start()
        try:
            yield
        finally:
            # остановка
            hub.loop = None
            pipeline.stop()
            for src in registry.sources.values():
                src.stop()

    app = FastAPI(title="Peleng", lifespan=lifespan)
    app.state.registry = registry
    app.state.pipeline = pipeline
    app.state.hub = hub
    app.state.board = board

    def _replay_loop(path: str):
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        msgs = [json.loads(x) for x in lines if x.strip()]
        head = next((m for m in msgs if m.get("type") == "hello"), None)
        frames = [m for m in msgs if m.get("type") == "frame"]
        if head:
            state["replay_hello"] = head
        print(f"[replay] {len(frames)} кадров из {path}")
        while True:
            if not frames:
                time.sleep(1)
                continue
            t_prev = frames[0]["t"]
            for m in frames:
                time.sleep(max(0.0, min(1.0, m["t"] - t_prev)))
                t_prev = m["t"]
                state["replay_t"] = m["t"]
                hub.publish(m)

    # ------------------------------------------------------------------ страницы
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    async def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/cam")
    async def cam_page():
        return FileResponse(WEB_DIR / "cam.html")

    # ------------------------------------------------------------------ WebSocket
    @app.websocket("/ws/stream")
    async def ws_stream(ws: WebSocket):
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        hub.queues.add(q)
        try:
            h = state.get("replay_hello") if replay else None
            await ws.send_text(json.dumps(h or hello(), ensure_ascii=False))
            if state["room_points"]:
                await ws.send_text(json.dumps(state["room_points"]))
            while True:
                text = await q.get()
                await ws.send_text(text)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.queues.discard(q)

    @app.websocket("/ws/push/{cam_id}")
    async def ws_push(ws: WebSocket, cam_id: str):
        await ws.accept()
        src = registry.sources.get(cam_id)
        if not isinstance(src, PushSource):
            src = PushSource(cam_id)
            registry.add(src)
        print(f"[{cam_id}] телефон подключился")
        hub.publish({"type": "cameras", "cameras": registry.describe()})
        loop = asyncio.get_running_loop()
        send_lock = asyncio.Lock()

        async def send(m: dict) -> None:
            try:
                async with send_lock:
                    await ws.send_text(json.dumps(m))
            except (WebSocketDisconnect, RuntimeError):
                pass

        def sender(m: dict) -> None:  # вызывается из потока конвейера
            asyncio.run_coroutine_threadsafe(send(m), loop)

        src.sender = sender
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    data = msg["bytes"]
                    if len(data) < 9 or data[8:10] != b"\xff\xd8":
                        continue
                    (ts,) = struct.unpack("<d", data[:8])
                    # защита от грубо неверного времени (часы не синхронизированы)
                    if abs(ts - time.time()) > 2.0:
                        ts = time.time()
                    # JPEG не распаковываем здесь: это блокировало бы цикл событий (пинги, другие телефоны)
                    src.push_jpeg(np.frombuffer(data, np.uint8, offset=8), ts)
                elif msg.get("text"):
                    m = json.loads(msg["text"])
                    if m.get("type") == "ping":
                        await send({"type": "pong", "tc": m.get("tc"), "ts": time.time()})
                    elif m.get("type") == "hello":
                        src.name = m.get("name") or cam_id
                        src.info = {k: v for k, v in m.items() if k not in ("type", "name")}
                        if m.get("hfov_deg"):
                            registry.hfov[cam_id] = float(m["hfov_deg"])
        except WebSocketDisconnect:
            pass
        finally:
            if src.sender is sender:
                src.sender = None
            print(f"[{cam_id}] телефон отключился")
            hub.publish({"type": "cameras", "cameras": registry.describe()})

    # ------------------------------------------------------------------ состояние
    @app.get("/api/state")
    async def api_state():
        return {**hello(), "pipeline": pipeline.stats, "intrinsics_jobs": state["intr_jobs"]}

    def _src(cam_id: str):
        src = registry.sources.get(cam_id)
        if src is None:
            raise HTTPException(404, f"нет камеры {cam_id}")
        return src

    def _overlay(cam_id: str, img: np.ndarray, show_board: bool) -> np.ndarray:
        img = img.copy()
        ts_dets = pipeline.last_dets.get(cam_id)
        if ts_dets:
            _, persons, det_w = ts_dets
            s = img.shape[1] / det_w  # детекции в пикселях кадра, на котором их нашли
            for p in persons:
                x1, y1, x2, y2 = (p.bbox * s).astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 2)
                kp = (p.kp * s).astype(int)
                for i, j in SKELETON:
                    if p.kc[i] > 0.35 and p.kc[j] > 0.35:
                        cv2.line(img, tuple(kp[i]), tuple(kp[j]), (0, 255, 0), 2)
        if show_board:
            det = board.detect(img)
            if det is not None:
                for c in det.corners.astype(int):
                    cv2.circle(img, tuple(c), 4, (255, 0, 255), -1)
                cv2.putText(img, f"board: {len(det.ids)} corners", (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        m = registry.model(cam_id)
        if m is not None and m.calibrated:
            m = m.resized((img.shape[1], img.shape[0]))  # превью меньше кадра
            # оси мира в начале координат
            pts = np.array([[0, 0, 0], [0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]], float)
            if (m.depth(pts) > 0.05).all():
                px = m.project(pts).astype(int)
                for k, col in zip((1, 2, 3), ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
                    cv2.line(img, tuple(px[0]), tuple(px[k]), col, 3)
        return img

    # превью: каждый кадр сжимается один раз на вариант (камера, оверлеи, ширина) и раздаётся всем зрителям
    preview_cache: dict[tuple, tuple[int, bytes]] = {}

    def _preview_jpeg(cam_id: str, f, overlay: bool, show_board: bool, width: int) -> bytes:
        key = (cam_id, overlay, show_board, width)
        hit = preview_cache.get(key)
        if hit is not None and hit[0] == f.seq:
            return hit[1]
        img = f.image
        if width and img.shape[1] > width:
            # уменьшаем до оверлея: рисовать и искать доску на маленьком кадре дешевле
            img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
        if overlay:
            img = _overlay(cam_id, img, show_board)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        data = buf.tobytes()
        preview_cache[key] = (f.seq, data)
        return data

    @app.get("/api/cameras/{cam_id}/snapshot.jpg")
    async def snapshot(cam_id: str, overlay: int = 1, board_overlay: int = 0, width: int = 640):
        f = _src(cam_id).latest()
        if f is None:
            raise HTTPException(503, "кадров ещё нет")
        data = await asyncio.to_thread(_preview_jpeg, cam_id, f, bool(overlay), bool(board_overlay), width)
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/cameras/{cam_id}/mjpeg")
    async def mjpeg(cam_id: str, board_overlay: int = 0, width: int = 640, fps: float = 5.0):
        src = _src(cam_id)

        async def gen():
            # пока показываем найденную доску — кадры в полном разрешении, как при калибровке
            with pipeline.full_res([cam_id]) if board_overlay else contextlib.nullcontext():
                last = -1
                while True:
                    f = src.latest()
                    if f is not None and f.seq != last:
                        last = f.seq
                        data = await asyncio.to_thread(_preview_jpeg, cam_id, f, True, bool(board_overlay), width)
                        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
                    await asyncio.sleep(1.0 / fps)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/board.png")
    async def board_png(px_per_square: int = 200):
        ok, buf = cv2.imencode(".png", board.image(px_per_square))
        return Response(buf.tobytes(), media_type="image/png")

    # ------------------------------------------------------------------ калибровка
    def _collect_board(cam_ids: list[str], duration: float = 1.2) -> dict[str, list]:
        dets: dict[str, list] = {c: [] for c in cam_ids}
        seen: dict[str, int] = {}
        with pipeline.full_res(cam_ids):
            t_end = time.time() + FULL_RES_SWITCH_S + duration
            while time.time() < t_end:
                for c in cam_ids:
                    src = registry.sources[c]
                    f = src.latest()
                    if f is None or seen.get(c) == f.seq:
                        continue
                    seen[c] = f.seq
                    if not src.is_full_res(f):  # кадр, запрошенный ещё до переключения
                        continue
                    d = board.detect(f.image)
                    if d is not None:
                        dets[c].append((f, d))
                time.sleep(0.03)
        return dets

    def _do_extrinsics(cam_ids: list[str] | None) -> dict:
        ids = [c for c in (cam_ids or list(registry.sources)) if c in registry.sources and registry.sources[c].online]
        if not ids:
            return {"ok": False, "message": "нет активных камер", "cameras": {}}
        collected = _collect_board(ids)
        report = {}
        for c in ids:
            items = collected[c]
            if not items:
                report[c] = {"ok": False, "message": "доска не видна"}
                continue
            f0 = items[-1][0]
            h, w = f0.image.shape[:2]
            m = registry.model(c, (w, h))
            refine = cfg["pipeline"].get("refine_focal_on_extrinsics", False) and m.meta.get("intrinsics") not in ("board", "demo")
            res = solve_extrinsics(board, m, [d for _, d in items], refine_f=refine)
            if not res.ok:
                report[c] = {"ok": False, "message": res.message}
                continue
            K = res.K if res.K is not None else m.K
            meta = dict(m.meta)
            if res.K is not None:
                meta["intrinsics"] = "approx+focal"
            meta.update({"extrinsics_reproj_px": round(res.reproj_px, 2), "extrinsics_time": time.time(), "extrinsics_points": res.n_points})
            new = CameraModel(K, m.dist, m.size, res.R, res.t, meta)
            registry.set_model(c, new)
            report[c] = {
                "ok": True,
                "reproj_px": round(res.reproj_px, 2),
                "points": res.n_points,
                "frames": len(items),
                "center": np.round(new.center, 3).tolist(),
                "dist_to_origin_m": round(float(np.linalg.norm(new.center)), 2),
                "hfov_deg": round(new.hfov_deg(), 1),
                "focal_refined": res.K is not None,
            }
        pipeline.reset_tracking()
        hub.publish({"type": "cameras", "cameras": registry.describe()})
        return {"ok": any(r["ok"] for r in report.values()), "cameras": report}

    @app.post("/api/calib/extrinsics")
    async def api_extrinsics(body: dict | None = None):
        cam_ids = (body or {}).get("cameras")
        return await asyncio.to_thread(_do_extrinsics, cam_ids)

    def _intrinsics_job(cam_id: str, n: int, interval: float):
        with pipeline.full_res([cam_id]):
            _intrinsics_collect(cam_id, n, interval)

    def _intrinsics_collect(cam_id: str, n: int, interval: float):
        job = state["intr_jobs"][cam_id]
        dets = []
        last_corners = None
        seen = -1
        size = None
        t_end = time.time() + max(60.0, n * interval * 6)
        while len(dets) < n and time.time() < t_end and job["state"] == "collecting":
            src = registry.sources[cam_id]
            f = src.latest()
            if f is None or f.seq == seen:
                time.sleep(0.05)
                continue
            seen = f.seq
            if not src.is_full_res(f):
                continue
            d = board.detect(f.image)
            if d is not None and len(d.ids) >= 8:
                # пропускаем почти одинаковые кадры — нужны разные ракурсы
                if last_corners is None or _moved(last_corners, d) > 0.04 * f.image.shape[1]:
                    dets.append(d)
                    last_corners = d
                    size = (f.image.shape[1], f.image.shape[0])
                    job["n"] = len(dets)
                    time.sleep(interval)
            time.sleep(0.02)
        if job["state"] != "collecting":
            return
        try:
            K, dist, rms = calibrate_intrinsics(dets, size)
            old = registry.model(cam_id)
            meta = {"intrinsics": "board", "intrinsics_rms": round(rms, 3), "intrinsics_views": len(dets)}
            # модель могла быть пересчитана под уменьшенные кадры трекинга — сравниваем пропорции, а не размер
            if old is not None and old.calibrated and abs(old.size[0] / old.size[1] - size[0] / size[1]) < 0.01:
                new = CameraModel(K, dist, size, old.R, old.t, {**old.meta, **meta})
            else:
                new = CameraModel(K, dist, size, meta=meta)
            registry.set_model(cam_id, new)
            job.update(state="done", rms=round(rms, 3), hfov_deg=round(new.hfov_deg(), 1))
            hub.publish({"type": "cameras", "cameras": registry.describe()})
        except Exception as e:  # noqa: BLE001
            job.update(state="error", message=str(e))

    @app.post("/api/calib/{cam_id}/intrinsics/start")
    async def api_intr_start(cam_id: str, body: dict | None = None):
        _src(cam_id)
        body = body or {}
        n = int(body.get("n", 15))
        state["intr_jobs"][cam_id] = {"state": "collecting", "n": 0, "need": n}
        threading.Thread(target=_intrinsics_job, args=(cam_id, n, float(body.get("interval", 0.3))), daemon=True).start()
        return state["intr_jobs"][cam_id]

    @app.post("/api/calib/{cam_id}/intrinsics/cancel")
    async def api_intr_cancel(cam_id: str):
        job = state["intr_jobs"].get(cam_id)
        if job:
            job["state"] = "cancelled"
        return job or {}

    @app.get("/api/calib/{cam_id}/intrinsics")
    async def api_intr_status(cam_id: str):
        return state["intr_jobs"].get(cam_id, {"state": "idle"})

    @app.post("/api/calib/{cam_id}/reset")
    async def api_reset(cam_id: str, body: dict | None = None):
        registry.reset(cam_id, keep_intrinsics=bool((body or {}).get("keep_intrinsics", True)))
        hub.publish({"type": "cameras", "cameras": registry.describe()})
        return {"ok": True}

    @app.post("/api/calib/{cam_id}/hfov")
    async def api_hfov(cam_id: str, body: dict):
        """Задать угол обзора вручную (если калибровки оптики нет)."""
        registry.hfov[cam_id] = float(body["hfov_deg"])
        m = registry.model(cam_id)
        if m is not None and m.meta.get("intrinsics", "approx") != "board":
            registry.models.pop(cam_id, None)
            p = registry.calib_dir / f"{cam_id}.json"
            if p.exists():
                p.unlink()
        hub.publish({"type": "cameras", "cameras": registry.describe()})
        return {"ok": True}

    # ------------------------------------------------------------------ комната, запись
    def _latest_full_res(src, timeout: float = 1.5):
        t_end = time.time() + timeout
        while time.time() < t_end:
            f = src.latest()
            if f is not None and src.is_full_res(f):
                return f
            time.sleep(0.03)
        return src.latest()

    def _do_scan() -> dict:
        images, cams = {}, {}
        with pipeline.full_res():
            frames = {c: _latest_full_res(src) for c, src in list(registry.sources.items())}
        for c, f in frames.items():
            if f is None:
                continue
            m = registry.model(c, (f.image.shape[1], f.image.shape[0]))
            if m is not None and m.calibrated:
                images[c], cams[c] = f.image, m
        if len(images) < 2:
            return {"type": "room_points", "points": [], "colors": [], "pairs": [], "message": "нужно ≥ 2 откалиброванных камер"}
        res = scan_points(cams, images)
        state["room_points"] = res
        hub.publish(res)
        return res

    @app.post("/api/room/scan")
    async def api_scan():
        res = await asyncio.to_thread(_do_scan)
        return {"points": len(res["points"]), "pairs": res["pairs"], "message": res.get("message", "ok")}

    @app.post("/api/room/clear")
    async def api_scan_clear():
        state["room_points"] = None
        hub.publish({"type": "room_points", "points": [], "colors": [], "pairs": []})
        return {"ok": True}

    @app.post("/api/record/start")
    async def api_rec_start():
        return {"file": pipeline.start_recording()}

    @app.post("/api/record/stop")
    async def api_rec_stop():
        return {"file": pipeline.stop_recording()}

    @app.post("/api/tracks/reset")
    async def api_tracks_reset():
        pipeline.reset_tracking()
        return {"ok": True}

    @app.get("/api/demo/truth")
    async def api_truth():
        if demo_world is None:
            raise HTTPException(404, "только в демо-режиме")
        return JSONResponse({"t": time.time(), "people": [np.round(p, 3).tolist() for p, _ in demo_world.people(time.time())]})

    # ------------------------------------------------------------------ расширения
    @app.get("/api/ext")
    async def api_ext():
        """Скрипты интерфейса расширений (web/ext/*.js) — вьювер подключает их сам."""
        d = WEB_DIR / "ext"
        return sorted(p.name for p in d.glob("*.js")) if d.is_dir() else []

    state["extensions"] = load_extensions(app, ExtContext(cfg, registry, pipeline, hub, board, state, data_dir))
    if state["extensions"]:
        print(f"[ext] расширения: {', '.join(state['extensions'])}")

    return app


def _moved(a, b) -> float:
    common, ia, ib = np.intersect1d(a.ids, b.ids, return_indices=True)
    if len(common) < 4:
        return 1e9
    return float(np.linalg.norm(a.corners[ia] - b.corners[ib], axis=1).mean())
