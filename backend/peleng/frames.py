"""Запись кадров камер рядом с сессией (чтобы при --replay во вьювере были превью).

Формат: data/sessions/session_X.jsonl  +  data/sessions/session_X_frames/
    index.jsonl             — по строке на кадр: {"cam": "iphone15", "t": 1730000000.1234, "file": "iphone15/1730000000123.jpg"}
    <камера>/<t в мс>.jpg   — уменьшенные кадры (по умолчанию ширина ≤ 480 px, JPEG quality 70, ≤ 5 к/с на камеру)
t — время кадра в часах сервера (Frame.ts), то же, что у сообщений "frame" в .jsonl.

JPEG-файлы, а не видео: cv2.VideoWriter на Windows зависит от кодеков, а файлы читаются где угодно и любой кадр доступен сразу.

Кодирование и запись идут в отдельном потоке с ограниченной очередью: конвейер только кладёт ссылку на кадр
(это дёшево), а при переполнении кадры отбрасываются — конвейер не ждёт диск.
"""
from __future__ import annotations

import bisect
import json
import queue
import re
import threading
from pathlib import Path

import cv2
import numpy as np

DEFAULTS = {"enabled": True, "max_width": 480, "fps": 5.0, "quality": 70, "queue": 64}
INDEX_NAME = "index.jsonl"


def settings(cfg: dict) -> dict:
    return {**DEFAULTS, **((cfg.get("recording") or {}).get("frames") or {})}


def frames_dir_for(session_path: str | Path) -> Path:
    p = Path(session_path)
    return p.with_name(p.stem + "_frames")


def _safe(cam_id: str) -> str:
    return re.sub(r"[^\w.\-]", "_", cam_id) or "cam"


class FrameWriter:
    """Пишет кадры одной сессии. offer() — из потока конвейера, остальное делает свой поток."""

    def __init__(self, root: Path, s: dict):
        self.root = Path(root)
        self.max_width = int(s["max_width"])
        self.quality = int(s["quality"])
        self.min_dt = 1.0 / max(float(s["fps"]), 0.1) - 0.002
        self._q: queue.Queue = queue.Queue(maxsize=max(1, int(s["queue"])))
        self._last: dict[str, float] = {}
        self.saved = 0
        self.dropped = 0
        self._thread = threading.Thread(target=self._run, daemon=True, name="frames-writer")
        self._thread.start()

    def offer(self, cam_id: str, ts: float, image: np.ndarray | None) -> None:
        if image is None or ts - self._last.get(cam_id, -1e18) < self.min_dt:
            return
        self._last[cam_id] = ts
        try:
            self._q.put_nowait((cam_id, ts, image))
        except queue.Full:
            self.dropped += 1  # диск/процессор не успевают — пропускаем кадр, а не тормозим конвейер

    def close(self, timeout: float = 5.0) -> None:
        try:
            self._q.put(None, timeout=timeout)
        except queue.Full:
            return
        self._thread.join(timeout)

    def _run(self) -> None:
        index = None
        try:
            while True:
                item = self._q.get()
                if item is None:
                    break
                cam_id, ts, img = item
                try:
                    if img.shape[1] > self.max_width:
                        img = cv2.resize(img, (self.max_width, max(1, round(img.shape[0] * self.max_width / img.shape[1]))), interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                    if not ok:
                        continue
                    rel = f"{_safe(cam_id)}/{int(round(ts * 1000))}.jpg"
                    f = self.root / rel
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_bytes(buf.tobytes())
                    if index is None:
                        index = open(self.root / INDEX_NAME, "a", encoding="utf-8")
                    index.write(json.dumps({"cam": cam_id, "t": round(ts, 4), "file": rel}, ensure_ascii=False) + "\n")
                    index.flush()
                    self.saved += 1
                except Exception as e:  # noqa: BLE001 — сбой записи кадра не должен убивать поток
                    print(f"[frames] не записан кадр {cam_id}: {e!r}")
        finally:
            if index:
                index.close()


class FrameIndex:
    """Индекс записанных кадров для replay: поиск кадра камеры, ближайшего по времени."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.times: dict[str, list[float]] = {}
        self.files: dict[str, list[str]] = {}
        p = self.root / INDEX_NAME
        if not p.is_file():
            return
        rows = []
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(line)
                rows.append((str(r["cam"]), float(r["t"]), str(r["file"])))
            except (ValueError, KeyError, TypeError):  # оборванная последняя строка и т. п.
                continue
        for cam, t, file in sorted(rows, key=lambda r: (r[0], r[1])):
            self.times.setdefault(cam, []).append(t)
            self.files.setdefault(cam, []).append(file)

    @property
    def cams(self) -> list[str]:
        return sorted(self.times)

    def nearest(self, cam: str, t: float | None) -> int | None:
        ts = self.times.get(cam)
        if not ts:
            return None
        if t is None:
            return 0
        i = bisect.bisect_left(ts, t)
        if i == 0:
            return 0
        if i == len(ts):
            return len(ts) - 1
        return i if ts[i] - t < t - ts[i - 1] else i - 1

    def path(self, cam: str, i: int) -> Path:
        return self.root / self.files[cam][i]

    def fps(self, cam: str) -> float:
        ts = self.times.get(cam, [])
        return round((len(ts) - 1) / (ts[-1] - ts[0]), 1) if len(ts) > 1 and ts[-1] > ts[0] else 0.0
