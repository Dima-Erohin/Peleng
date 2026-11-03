"""Запись кадров камер вместе с сессией и показ их при replay (peleng/frames.py, peleng/ext/frames.py) — на демо, без камер."""
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peleng import frames as fr  # noqa: E402


def _cfg(**frames):
    from peleng.config import load_config

    cfg = load_config()
    cfg["demo"] = {"cameras": 2, "people": 1}
    cfg["recording"] = {"frames": frames}
    return cfg


def _record(tmp_path, seconds=2.0, **frames) -> Path:
    from fastapi.testclient import TestClient

    from peleng.server import create_app

    app = create_app(_cfg(**frames), demo=True, data_dir=tmp_path)
    with TestClient(app) as client:
        path = Path(client.post("/api/record/start").json()["file"])
        time.sleep(seconds)
        assert client.post("/api/record/stop").json()["file"] == str(path)
    return path


def _index(path: Path) -> list[dict]:
    idx = fr.frames_dir_for(path) / fr.INDEX_NAME
    return [json.loads(x) for x in idx.read_text(encoding="utf-8").splitlines()]


def test_writer_limits(tmp_path):
    """Прореживание по fps, ширина ≤ max_width, индекс {cam, t, file}, переполнение очереди не блокирует."""
    w = fr.FrameWriter(tmp_path / "f", {**fr.DEFAULTS, "fps": 5, "max_width": 200})
    img = np.full((540, 960, 3), 90, np.uint8)
    t0 = 1000.0
    for i in range(60):  # 4 с по 15 к/с
        w.offer("cam/1", t0 + i / 15, img)  # "/" в id камеры не должен ломать путь
    w.close()
    rows = [json.loads(x) for x in (tmp_path / "f" / fr.INDEX_NAME).read_text().splitlines()]
    assert 18 <= len(rows) <= 21 and w.saved == len(rows)
    assert all(r["cam"] == "cam/1" for r in rows)
    assert all(b["t"] - a["t"] >= 0.19 for a, b in zip(rows, rows[1:]))
    out = cv2.imread(str(tmp_path / "f" / rows[0]["file"]))
    assert out.shape[1] == 200 and out.shape[0] == 112
    idx = fr.FrameIndex(tmp_path / "f")
    assert idx.cams == ["cam/1"]
    assert idx.nearest("cam/1", t0 + 1.01) == min(range(len(rows)), key=lambda i: abs(rows[i]["t"] - (t0 + 1.01)))
    assert idx.nearest("нет", 1.0) is None

    # очередь на 1 кадр и «медленный» писатель: offer не блокируется, лишнее отбрасывается
    w2 = fr.FrameWriter(tmp_path / "g", {**fr.DEFAULTS, "fps": 1000, "queue": 1})
    big = np.random.randint(0, 255, (1080, 1920, 3), np.uint8)
    t = time.time()
    for i in range(50):
        w2.offer("a", i, big)
    assert time.time() - t < 0.5
    assert w2.dropped > 0
    w2.close()


def test_record_demo_session_has_frames(tmp_path):
    secs = 3.0
    path = _record(tmp_path, seconds=secs, fps=4, max_width=320, quality=60)
    rows = _index(path)
    assert {r["cam"] for r in rows} == {"demo1", "demo2"}
    for cam in ("demo1", "demo2"):
        ts = [r["t"] for r in rows if r["cam"] == cam]
        assert 3 <= len(ts) <= 4 * (secs + 1) + 1
        assert all(b - a >= 0.24 for a, b in zip(ts, ts[1:]))
    for r in rows:
        img = cv2.imread(str(fr.frames_dir_for(path) / r["file"]))
        assert img is not None and img.shape[1] <= 320
    # время кадров укладывается в интервал записанных сообщений frame
    frames = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()][1:]
    assert min(r["t"] for r in rows) >= frames[0]["t"] - 0.5 and max(r["t"] for r in rows) <= frames[-1]["t"] + 0.5


def test_disabled(tmp_path):
    path = _record(tmp_path, seconds=1.0, enabled=False)
    assert not fr.frames_dir_for(path).exists()


def test_replay_serves_recorded_frames(tmp_path):
    from fastapi.testclient import TestClient

    from peleng.server import create_app

    path = _record(tmp_path / "rec", seconds=3.0)
    app = create_app(_cfg(), replay=str(path), data_dir=tmp_path / "rep")
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            assert ws.receive_json()["type"] == "hello"
            for _ in range(5):
                assert ws.receive_json()["type"] == "frame"
        cams = {c["id"]: c for c in client.get("/api/state").json()["cameras"]}
        assert set(cams) == {"demo1", "demo2"} and all(c["online"] and c["calibrated"] for c in cams.values())
        r = client.get("/api/cameras/demo1/snapshot.jpg?overlay=0&width=320")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[1] <= 320
        assert client.get("/api/cameras/demo2/snapshot.jpg").status_code == 200
        assert client.get("/api/cameras/нет/snapshot.jpg").status_code == 404


def test_replay_nearest_frame_follows_time(tmp_path):
    """ReplaySource.latest() следует за state["replay_t"]."""
    from peleng.ext.frames import ReplaySource

    root = tmp_path / "s_frames"
    w = fr.FrameWriter(root, {**fr.DEFAULTS, "fps": 100})
    for i, v in enumerate((10, 120, 240)):
        w.offer("c", 100.0 + i, np.full((48, 64, 3), v, np.uint8))
    w.close()
    state = {"replay_t": None}
    src = ReplaySource("c", None, fr.FrameIndex(root), state)
    assert abs(src.latest().ts - 100.0) < 1e-3
    state["replay_t"] = 101.4
    assert abs(src.latest().ts - 101.0) < 1e-3
    state["replay_t"] = 500
    f = src.latest()
    assert abs(f.ts - 102.0) < 1e-3 and f.image[0, 0].mean() > 200
    assert src.latest() is f  # кадр кэшируется


def test_replay_without_frames_dir_and_trajectory(tmp_path):
    """Старая запись (без папки кадров) работает как раньше; папка кадров не мешает списку сессий траекторий."""
    from fastapi.testclient import TestClient

    from peleng.server import create_app

    path = _record(tmp_path / "rec", seconds=2.0)
    # trajectory: в списке только .jsonl, восстановление не падает с соседней папкой кадров
    app = create_app(_cfg(), demo=True, data_dir=tmp_path / "rec")
    with TestClient(app) as client:
        names = [s["name"] for s in client.get("/api/trajectory/sessions").json()]
        assert names == [path.name]
        assert client.get("/api/trajectory/restore", params={"session": path.name}).status_code == 200

    old = tmp_path / "old" / path.name
    old.parent.mkdir()
    shutil.copy(path, old)
    app = create_app(_cfg(), replay=str(old), data_dir=tmp_path / "rep")
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            assert ws.receive_json()["type"] == "frame"
        assert client.get("/api/state").status_code == 200
        assert client.get("/api/cameras/demo1/snapshot.jpg").status_code == 404


def test_replay_does_not_touch_real_calibration(tmp_path):
    """Кнопка «Сброс» камеры при replay не удаляет настоящий data/calib/<id>.json."""
    from fastapi.testclient import TestClient

    from peleng.server import create_app

    path = _record(tmp_path / "rec", seconds=2.0)
    app = create_app(_cfg(), replay=str(path), data_dir=tmp_path / "rep")
    calib = tmp_path / "rep" / "calib" / "demo1.json"
    calib.write_text("{}")
    with TestClient(app) as client:
        client.post("/api/calib/demo1/reset", json={})
        client.post("/api/calib/demo1/hfov", json={"hfov_deg": 70})
    assert calib.exists()
