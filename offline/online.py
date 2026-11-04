"""Онлайн-прогон: видео камер играют как живые потоки, грузовик считается в момент показа.

    python offline/online.py --data <папка с mp4 и csv камер> [--weights models/truck_pose_All.pt] [--port 8600]

Как устроено:
* у каждой камеры свой поток чтения: видео декодируется по реальным часам (кадр k — в момент t0 + k/fps),
  как с живой камеры; в памяти — несколько последних кадров;
* рабочий цикл берёт самый свежий кадр, который есть у всех камер, прогоняет детектор позы грузовика
  (YOLO-pose, наши веса) по всем камерам одним вызовом, при пропуске — сильную YOLO в окне вокруг прогноза,
  затем шаг 3D-трекера (фильтр Калмана, тот же, что в offline/run.py, пресет truck_fused). Кадры, пришедшие,
  пока сеть занята, пропускаются — точка каждого шага посчитана только по кадрам <= k (как в живой системе);
* каждая точка сразу уходит по WebSocket /ws/online во фронтенд (страница «Траектории»), где траектория
  дорисовывается под идущее видео.

Фронтенд — собранный frontend (backend/web/build), страница открывается на http://localhost:<port>/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import webbrowser
from collections import OrderedDict
from pathlib import Path

import numpy as np
from fastapi import WebSocket  # на уровне модуля: FastAPI разрешает аннотации по глобальным именам

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "offline"

from .tracker3d import Tracker3D, TrackerConfig  # noqa: E402
from .ue_data import load_dataset, world_to_ue  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "backend" / "web"
DEFAULT_WEIGHTS = ROOT / "models" / "truck_pose_All.pt"


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


class LiveVideo(threading.Thread):
    """Видео как живая камера: кадр k «снимается» в момент t0 + k/fps.

    Все кадры проматываются по часам (grab — без перевода в картинку, с аппаратным декодированием, если есть),
    в картинку распаковывается только кадр, который запросил рабочий цикл (request) — как камера, у которой
    забирают свежий кадр, когда сеть освободилась."""

    def __init__(self, path: Path, fps: float, t0: float):
        super().__init__(daemon=True, name=f"video-{Path(path).stem}")
        self.path, self.fps, self.t0 = path, fps, t0
        self.frames: OrderedDict[int, np.ndarray] = OrderedDict()
        self.latest = -1
        self.want: int | None = None
        self.ended = False
        self.running = True
        self.lock = threading.Lock()
        self.ready = threading.Condition(self.lock)

    def _open(self):
        import cv2

        cap = cv2.VideoCapture(str(self.path), cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
        if not cap.isOpened():
            cap = cv2.VideoCapture(str(self.path))
        return cap

    def run(self):
        cap = self._open()
        k = -1
        while self.running:
            if k >= int((time.time() - self.t0) * self.fps):
                time.sleep(0.003)
                continue
            if not cap.grab():
                break
            k += 1
            w = self.want
            img = None
            if w is not None and k >= w:
                ok, img = cap.retrieve()
                img = img if ok else None
            with self.lock:
                if img is not None:
                    self.frames[k] = img
                    while len(self.frames) > 4:
                        self.frames.popitem(last=False)
                    self.want = None
                self.latest = k
                self.ready.notify_all()
        with self.lock:
            self.ended = True
            self.ready.notify_all()
        cap.release()

    def request(self, k: int) -> None:
        with self.lock:
            self.want = k

    def wait(self, k: int, deadline: float):
        """Кадр с номером >= k (номер, картинка) или (None, None), если видео кончилось / не успели."""
        with self.lock:
            while self.running:
                got = [j for j in self.frames if j >= k]
                if got:
                    return got[0], self.frames[got[0]]
                left = deadline - time.time()
                if self.ended or left <= 0:
                    return None, None
                self.ready.wait(left)
        return None, None

    def stop(self):
        self.running = False


class OnlineSession:
    """Один прогон от начала видео: потоки камер + рабочий цикл детекции и трекинга."""

    def __init__(self, engine: "OnlineEngine", start_delay: float = 1.5):
        self.engine = engine
        ds = engine.ds
        self.t0 = time.time() + start_delay
        self.id = int(self.t0 * 1000)
        self.videos = [LiveVideo(c.video_path, ds.fps, self.t0) for c in ds.cameras]
        self.points: list[dict] = []
        self.running = True
        self.done = False
        self.worker = threading.Thread(target=self._loop, daemon=True, name="online-worker")

    def start(self):
        for v in self.videos:
            v.start()
        self.worker.start()

    def stop(self):
        self.running = False
        for v in self.videos:
            v.stop()

    def _loop(self):
        eng, ds = self.engine, self.engine.ds
        tracker = Tracker3D(TrackerConfig(**eng.tracker_kw))
        ids = [c.cam_id for c in ds.cameras]
        last = -1
        n = ds.n_frames
        while self.running:
            # следующий кадр по часам: его и снимают все камеры (кадры, пока сеть была занята, пропущены)
            now_k = int((time.time() - self.t0) * ds.fps) + 1
            if now_k <= last:
                time.sleep(0.003)
                continue
            if now_k >= n or all(v.ended for v in self.videos):
                break
            for v in self.videos:
                v.request(now_k)
            deadline = self.t0 + now_k / ds.fps + 0.5
            got = [v.wait(now_k, deadline) for v in self.videos]
            ks = [g[0] for g in got if g[0] is not None]
            if not ks:
                if all(v.ended for v in self.videos):
                    break
                continue
            k = min(ks)
            imgs = [image if frame == k else None for frame, image in got]
            t_start = self.t0 + k / ds.fps  # момент «съёмки» кадра — от него считаем задержку
            t_proc = time.time()
            cams = [c.model(k) for c in ds.cameras]
            row = eng.detect(k, imgs, cams, tracker, ids)
            t_det = time.time()
            r = tracker.step(k, cams, row, ids)
            msg = {"type": "point", "session": self.id, "frame": k, "t": k / ds.fps, "mode": r.mode,
                   "limited_update": bool(r.info.get("limited_update")),
                   "n_cams": int(r.n_cams), "cams": [int(c) for c in (r.cams or [])],
                   "used_det": r.info.get("used_det", []),
                   "proc_ms": round((time.time() - t_proc) * 1000), "detect_ms": round((t_det - t_proc) * 1000),
                   "lag_ms": round((time.time() - t_start) * 1000),
                   "dets": [[[round(v, 1) for v in d.bbox] + [round(d.score, 2), d.source] for d in cam] for cam in row]}
            if r.pos is not None:
                msg["pos"] = [round(float(v), 2) for v in world_to_ue(r.pos[None])[0]]  # Unreal, см
                if r.vel is not None:
                    v = np.array([r.vel[0], -r.vel[1], r.vel[2]]) * 100.0  # м/с -> см/с, ось Y Unreal
                    msg["vel"] = [round(float(x), 1) for x in v]
                msg["sigma_m"] = round(float(r.sigma_m), 2) if np.isfinite(r.sigma_m) else None
            self.points.append(msg)
            eng.broadcast(msg)
            last = k
        self.done = True
        eng.broadcast({"type": "done", "session": self.id, "points": len(self.points)})


class OnlineEngine:
    def __init__(self, data: str, weights: str, device: str, fallback: str, conf: float, roi_weights: str | None = None,
                 roi_below: int = 99):
        from .detectors.truck_pose import TruckPoseDetector
        from .presets import PRESETS

        self.ds = load_dataset(data)
        self.device = pick_device(device)
        self.weights = str(weights)
        print(f"детектор позы: {self.weights}, устройство {self.device}", flush=True)
        self.pose = TruckPoseDetector(self.weights, conf=conf, device=self.device)
        spec = PRESETS["truck_fused"]
        kw = dict(spec["tracker"])
        kw["fps"] = self.ds.fps
        # счётчики «N шагов подряд» в онлайне считаются по обработанным шагам, а шаги идут реже кадров
        kw["reinit_frames"] = 3
        self.tracker_kw = kw
        self.drop_border = spec["fusion"].get("drop_border_px", 3.0)
        self.roi_below = roi_below  # окна ищем, только если поза нашла грузовик меньше чем в стольких камерах
        self.refiner = None
        if fallback == "roi":
            from .roi_refine import RoiRefiner

            r = spec["roi"]
            # веса окна (yolo11m.pt) скачиваются один раз в models/ — заранее, а не посреди прогона
            roi_w = ROOT / "models" / (roi_weights or r["weights"])
            self.refiner = RoiRefiner(str(roi_w), imgsz=r["imgsz"], classes=tuple(r["classes"]), conf=r["conf"],
                                      device=self.device)
            self.refiner._load()
            self.refiner.model.predict(np.zeros((640, 640, 3), np.uint8), imgsz=r["imgsz"], verbose=False, device=self.device)
        self.clients: set = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.session: OnlineSession | None = None
        self.lock = threading.Lock()
        # прогрев сети, чтобы первый шаг не занял секунды
        blank = np.zeros((1080, 1920, 3), np.uint8)
        self.pose.detect_batch([blank] * len(self.ds.cameras))

    def detect(self, k: int, imgs: list, cams: list, tracker: Tracker3D, ids: list) -> list:
        from .roi_refine import merge

        pose = self.pose.detect_batch(imgs)
        row = []
        regs = tracker.predict_regions(cams, dt=(k - tracker.frame) / self.ds.fps) if (
            self.refiner is not None and tracker.x is not None and not tracker._lost()) else None
        need = []
        for i, (cam, img) in enumerate(zip(cams, imgs)):
            W, H = cam.size
            b = self.drop_border
            prim = [d for d in pose[i] if d.point_is_reliable(tracker.cfg.point)
                    and d.bbox[0] > b and d.bbox[1] > b and d.bbox[2] < W - b and d.bbox[3] < H - b]
            row.append(prim)
            if not prim and img is not None and regs is not None and regs[i] is not None:
                need.append(i)
        if need and sum(1 for d in row if d) < self.roi_below:
            # поза в этих камерах молчит, а трек есть: YOLO в окне вокруг прогноза, все окна одним вызовом
            found = self.refiner.detect_many([(imgs[i], self.refiner.window(regs[i], cams[i].size)) for i in need])
            for i, dets in zip(need, found):
                row[i] = merge([], dets)
        return row

    # --- сессии и рассылка ---
    def start(self) -> OnlineSession:
        with self.lock:
            if self.session is not None:
                self.session.stop()
            self.session = OnlineSession(self)
            self.session.start()
            s = self.session
        self.broadcast(self.hello())
        return s

    def hello(self) -> dict:
        s = self.session
        ds = self.ds
        return {"type": "hello", "server_time": time.time(), "fps": ds.fps, "n_frames": ds.n_frames,
                "sizes": [list(c.model(0).size) for c in ds.cameras],
                "cams": [c.cam_id for c in ds.cameras], "device": self.device, "weights": Path(self.weights).name,
                "fallback": "roi" if self.refiner else "none",
                "session": s.id if s else None, "t0": s.t0 if s else None, "done": s.done if s else None,
                "points": list(s.points) if s else []}

    def broadcast(self, msg: dict) -> None:
        if self.loop is None or not self.clients:
            return
        text = json.dumps(msg, ensure_ascii=False)
        for ws in list(self.clients):
            self.loop.call_soon_threadsafe(asyncio.ensure_future, self._send(ws, text))

    async def _send(self, ws, text: str) -> None:
        try:
            await ws.send_text(text)
        except Exception:  # noqa: BLE001
            self.clients.discard(ws)


def create_app(engine: OnlineEngine):
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    @asynccontextmanager
    async def lifespan(app):
        engine.loop = asyncio.get_running_loop()
        yield
        if engine.session:
            engine.session.stop()

    app = FastAPI(title="Peleng online", lifespan=lifespan)
    index = WEB_DIR / "build" / "index.html"

    @app.get("/")
    def root():
        if not index.exists():
            raise HTTPException(503, "Фронтенд не собран: cd frontend && npm ci && npm run build")
        return FileResponse(index)

    @app.get("/api/online/state")
    def state():
        h = engine.hello()
        h.pop("points")
        return JSONResponse(h)

    @app.post("/api/online/start")
    def start():
        return {"session": engine.start().id}

    @app.get("/api/online/video/{i}")
    def video(i: int):
        cams = engine.ds.cameras
        if not 0 <= i < len(cams) or cams[i].video_path is None:
            raise HTTPException(404)
        return FileResponse(cams[i].video_path, media_type="video/mp4")

    @app.websocket("/ws/online")
    async def ws_online(ws: WebSocket):
        await ws.accept()
        engine.clients.add(ws)
        try:
            # первый зритель запускает прогон; остальные подключаются к идущему
            if engine.session is None:
                engine.start()
            else:
                await ws.send_text(json.dumps(engine.hello(), ensure_ascii=False))
            while True:
                msg = json.loads(await ws.receive_text())
                if msg.get("type") == "start":
                    engine.start()
                elif msg.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong", "tc": msg.get("tc"), "server_time": time.time()}))
        except WebSocketDisconnect:
            pass
        finally:
            engine.clients.discard(ws)

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description="Онлайн-прогон: грузовик считается по видео в момент показа")
    ap.add_argument("--data", required=True, help="папка с Way1...CineCameraActorN.mp4 и object_transform_*.csv")
    ap.add_argument("--weights", default=os.environ.get("TRUCK_WEIGHTS") or str(DEFAULT_WEIGHTS),
                    help="веса детектора позы грузовика (по умолчанию models/truck_pose_All.pt)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda:0 | mps")
    ap.add_argument("--fallback", default="roi", choices=["roi", "none"],
                    help="roi — когда поза в камере молчит, искать грузовик yolo11m в окне вокруг прогноза")
    ap.add_argument("--roi-weights", default="yolo11s.pt",
                    help="модель окна: yolo11s.pt (по умолчанию) / yolo11m.pt (точнее, медленнее) / yolo11n.pt; качается в models/")
    ap.add_argument("--roi-below", type=int, default=99,
                    help="искать в окнах, только если поза нашла грузовик меньше чем в N камерах (2 — быстрее)")
    ap.add_argument("--conf", type=float, default=0.35, help="порог детектора позы")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--no-browser", action="store_true", help="не открывать браузер")
    a = ap.parse_args(argv)
    if not Path(a.weights).exists():
        sys.exit(f"нет весов {a.weights} — укажите --weights или TRUCK_WEIGHTS")
    engine = OnlineEngine(a.data, a.weights, a.device, a.fallback, a.conf, a.roi_weights, a.roi_below)
    app = create_app(engine)
    url = f"http://localhost:{a.port}/"
    print(f"\n=== Peleng online ===\n  откройте {url}  (воспроизведение; онлайн-расчёт — {url}?online=1)\n", flush=True)
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    import uvicorn

    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
