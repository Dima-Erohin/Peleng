"""Источники кадров и реестр камер.

* OpenCVSource — вебкамера ноутбука (индекс 0/1/…) или URL потока (IP Webcam, DroidCam: http://IP:PORT/video).
* PushSource — телефон открыл страницу /cam в браузере и шлёт JPEG по WebSocket (ничего не надо ставить).
  Кадры снимаются по команде сервера («capture»), см. Pipeline.request_frames и web/cam.html.

Время кадра — в часах сервера (time.time()). Телефон сам пересчитывает своё время в серверное
по пингам (см. web/cam.html), так что задержка сети не влияет на синхронизацию.
"""
from __future__ import annotations

import json
import platform
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .geometry import CameraModel, approx_K


class Frame:
    """Кадр. От телефона приходит JPEG — он распаковывается только при первом обращении к .image
    (большую часть кадров никто не читает, а распаковка в цикле событий тормозит пинги и синхронизацию)."""

    __slots__ = ("_image", "_jpeg", "_lock", "ts", "seq", "t_recv")

    def __init__(self, image: np.ndarray | None, ts: float, seq: int, jpeg: np.ndarray | None = None):
        self._image = image
        self._jpeg = jpeg
        self._lock = threading.Lock()
        self.ts = ts
        self.seq = seq
        self.t_recv = time.time()  # когда кадр пришёл на сервер

    @property
    def image(self) -> np.ndarray | None:
        if self._image is None and self._jpeg is not None:
            with self._lock:
                if self._image is None and self._jpeg is not None:
                    self._image = cv2.imdecode(self._jpeg, cv2.IMREAD_COLOR)
                    self._jpeg = None
        return self._image


class Source:
    kind = "base"

    def __init__(self, cam_id: str, name: str | None = None):
        self.id = cam_id
        self.name = name or cam_id
        self._frame: Frame | None = None
        self._seq = 0
        self._lock = threading.Lock()
        self._fps_t: list[float] = []
        self._hist: deque[Frame] = deque(maxlen=30)
        self.info: dict = {}

    def _put(self, image: np.ndarray | None, ts: float, jpeg: np.ndarray | None = None) -> None:
        with self._lock:
            self._seq += 1
            self._frame = Frame(image, ts, self._seq, jpeg)
            self._hist.append(self._frame)
            now = time.time()
            self._fps_t.append(now)
            self._fps_t = [x for x in self._fps_t if now - x < 2.0]

    def latest(self) -> Frame | None:
        with self._lock:
            return self._frame

    @property
    def fps(self) -> float:
        now = time.time()
        recent = [x for x in self._fps_t if now - x < 2.0]
        return len(recent) / 2.0

    @property
    def online(self) -> bool:
        f = self._frame
        return f is not None and time.time() - f.ts < 2.0

    def nearest(self, ts: float) -> Frame | None:
        """Кадр из недавней истории, ближайший по времени к ts (для камер-потоков рядом с кадрами по команде)."""
        with self._lock:
            hist = list(self._hist)
        return min(hist, key=lambda f: abs(f.ts - ts)) if hist else None

    on_demand = False

    def request(self, width: int) -> bool:
        return False

    def is_full_res(self, frame: Frame) -> bool:
        return True

    def start(self) -> None:  # noqa: B027
        pass

    def stop(self) -> None:  # noqa: B027
        pass


class OpenCVSource(Source):
    kind = "opencv"

    def __init__(self, cam_id: str, src, name: str | None = None, width: int | None = None, height: int | None = None):
        super().__init__(cam_id, name)
        self.src = src
        self.width = width
        self.height = height
        self._running = False
        self._thread: threading.Thread | None = None

    def _open(self):
        if isinstance(self.src, int) and platform.system() == "Windows":
            cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.src)
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self) -> None:
        while self._running:
            cap = self._open()
            if not cap.isOpened():
                print(f"[{self.id}] не удалось открыть источник {self.src!r}, повтор через 2 с")
                time.sleep(2)
                continue
            print(f"[{self.id}] источник {self.src!r} открыт")
            while self._running:
                ok, img = cap.read()
                if not ok:
                    print(f"[{self.id}] поток прервался, переподключение")
                    break
                self._put(img, time.time())
            cap.release()
            time.sleep(0.5)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"src-{self.id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False


class PushSource(Source):
    kind = "push"

    def __init__(self, cam_id: str, name: str | None = None):
        super().__init__(cam_id, name)
        # пока телефон подключён, сервер кладёт сюда функцию отправки сообщения ему (из любого потока)
        self.sender = None

    def push(self, image: np.ndarray, ts: float) -> None:
        self._put(image, ts)

    def push_jpeg(self, jpeg: np.ndarray, ts: float) -> None:
        self._put(None, ts, jpeg)

    @property
    def on_demand(self) -> bool:
        """Телефон снимает кадр по команде сервера, а не шлёт поток сам."""
        return self.sender is not None and bool(self.info.get("on_demand"))

    def request(self, width: int) -> bool:
        """Попросить телефон снять кадр. width — ширина кадра (0 — полное разрешение камеры)."""
        if not self.on_demand:
            return False
        self.sender({"type": "capture", "w": int(width)})
        return True

    def is_full_res(self, frame: Frame) -> bool:
        """Кадр в полном разрешении камеры (для калибровки нужны именно такие)."""
        full = max(self.info.get("vw") or 0, self.info.get("vh") or 0)
        img = frame.image
        return img is not None and max(img.shape[:2]) >= 0.95 * full


class CameraRegistry:
    """Камеры + их калибровка (хранится в data/calib/<id>.json)."""

    def __init__(self, calib_dir: Path, default_hfov_deg: float = 68.0):
        self.calib_dir = Path(calib_dir)
        self.calib_dir.mkdir(parents=True, exist_ok=True)
        self.default_hfov_deg = default_hfov_deg
        self.sources: dict[str, Source] = {}
        self.models: dict[str, CameraModel] = {}
        self.hfov: dict[str, float] = {}
        self._lock = threading.RLock()

    def add(self, source: Source, hfov_deg: float | None = None) -> None:
        with self._lock:
            self.sources[source.id] = source
            if hfov_deg:
                self.hfov[source.id] = hfov_deg
            path = self.calib_dir / f"{source.id}.json"
            if path.exists() and source.id not in self.models:
                try:
                    self.models[source.id] = CameraModel.from_dict(json.loads(path.read_text(encoding="utf-8")))
                    print(f"[{source.id}] загружена калибровка {path}")
                except Exception as e:  # noqa: BLE001
                    print(f"[{source.id}] не удалось прочитать калибровку: {e}")

    def remove(self, cam_id: str) -> None:
        with self._lock:
            src = self.sources.pop(cam_id, None)
            if src:
                src.stop()

    def model(self, cam_id: str, size: tuple[int, int] | None = None) -> CameraModel | None:
        """Модель камеры под текущее разрешение кадра (создаёт приблизительную, если калибровки нет)."""
        with self._lock:
            m = self.models.get(cam_id)
            if size is None:
                return m
            if m is None:
                hfov = self.hfov.get(cam_id, self.default_hfov_deg)
                m = CameraModel(approx_K(size[0], size[1], hfov), np.zeros(5), tuple(size), meta={"intrinsics": "approx", "hfov_deg": hfov})
                self.models[cam_id] = m
                return m
            if tuple(m.size) != tuple(size):
                m = m.resized(size)
                self.models[cam_id] = m
            return m

    def set_model(self, cam_id: str, model: CameraModel, save: bool = True) -> None:
        with self._lock:
            self.models[cam_id] = model
            if save:
                path = self.calib_dir / f"{cam_id}.json"
                path.write_text(json.dumps(model.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def reset(self, cam_id: str, keep_intrinsics: bool = True) -> None:
        with self._lock:
            m = self.models.get(cam_id)
            if m is None:
                return
            if keep_intrinsics and m.meta.get("intrinsics") == "board":
                meta = {k: v for k, v in m.meta.items() if not k.startswith("extr")}
                self.set_model(cam_id, CameraModel(m.K, m.dist, m.size, meta=meta))
            else:
                self.models.pop(cam_id, None)
                p = self.calib_dir / f"{cam_id}.json"
                if p.exists():
                    p.unlink()

    def describe(self) -> list[dict]:
        out = []
        with self._lock:
            for cid, src in self.sources.items():
                m = self.models.get(cid)
                d = {
                    "id": cid,
                    "name": src.name,
                    "kind": src.kind,
                    "online": src.online,
                    "fps": round(src.fps, 1),
                    "calibrated": bool(m and m.calibrated),
                    "info": src.info,
                }
                if m is not None:
                    d["size"] = list(m.size)
                    d["hfov_deg"] = round(m.hfov_deg(), 1)
                    d["intrinsics"] = m.meta.get("intrinsics", "approx")
                    d["K"] = [[round(v, 2) for v in row] for row in m.K.tolist()]
                    if m.calibrated:
                        d["R"] = [[round(v, 6) for v in row] for row in m.R.tolist()]
                        d["t"] = [round(v, 4) for v in m.t.reshape(-1).tolist()]
                        d["center"] = [round(v, 3) for v in m.center.tolist()]
                        d["extrinsics_reproj_px"] = m.meta.get("extrinsics_reproj_px")
                out.append(d)
        return out
