"""Отслеживание объектов на синтетике: без камер и без скачивания моделей."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peleng.geometry import CameraModel, approx_K, look_at  # noqa: E402
from peleng.objects import (  # noqa: E402
    TAG_DICT, ArucoObjectDetector, Det2D, ObjectTracker, associate_objects, mono_object, reconstruct_objects, synth_detections,
)

SIZE = (1280, 720)


def make_cam(center, target=(0, 0, 0.5), hfov=70.0):
    R, t = look_at(np.array(center, float), np.array(target, float))
    return CameraModel(approx_K(*SIZE, hfov), np.zeros(5), SIZE, R, t)


def three_cams():
    return {"a": make_cam([-2, -2, 2]), "b": make_cam([2, -1.5, 2.2]), "c": make_cam([0, 2.5, 2.0])}


def ball_dets(cams, X, cls="sports ball", r=0.1, rng=None, noise=0.0):
    return {c: synth_detections(m, [(cls, X, r)], rng, noise) for c, m in cams.items()}


def test_trajectory_three_cams():
    """Мяч летит по известной траектории, 3 камеры, шум 0.5 px -> ошибка 3D < 2 см."""
    cams = three_cams()
    tracker = ObjectTracker()
    rng = np.random.default_rng(1)
    errs_raw, errs_trk, ids = [], [], set()
    for k in range(90):
        t = k / 15
        X = np.array([0.8 * np.cos(0.8 * t), 0.6 * np.sin(0.8 * t), 0.2 + 0.5 * abs(np.sin(1.5 * t))])
        obs = reconstruct_objects(cams, ball_dets(cams, X, rng=rng, noise=0.5))
        assert len(obs) == 1 and obs[0].mode == "tri" and obs[0].cls == "sports ball"
        errs_raw.append(np.linalg.norm(obs[0].pos - X))
        out = tracker.update(t, obs)
        if k >= 10:
            assert len(out) == 1
            o = out[0]
            assert o["kind"] == "object" and o["cls"] == "sports ball" and o["mode"] == "tri" and len(o["cams"]) == 3
            ids.add(o["id"])
            errs_trk.append(np.linalg.norm(np.array(o["pos"]) - X))
    assert max(errs_raw) < 0.02, f"триангуляция: {max(errs_raw):.4f} м"
    assert np.median(errs_trk) < 0.02, f"после трекинга: {np.median(errs_trk):.4f} м"
    assert len(ids) == 1
    json.dumps(out)  # поле objects сериализуется


def test_two_same_class_objects_matched():
    """Два мяча: в каждой камере детекции в разном порядке — сопоставление по эпиполярам не путает их."""
    cams = three_cams()
    X1, X2 = np.array([0.4, 0.3, 0.1]), np.array([-0.5, -0.2, 0.1])
    rng = np.random.default_rng(2)
    dets = {}
    for i, (c, m) in enumerate(cams.items()):
        ds = synth_detections(m, [("sports ball", X1, 0.1), ("sports ball", X2, 0.1)], rng, 0.5)
        dets[c] = ds[::-1] if i % 2 else ds
    # посторонний объект другого класса там же, где мяч, — не мешает
    dets["a"].append(synth_detections(cams["a"], [("cup", X1, 0.05)])[0])
    clusters = associate_objects(cams, dets)
    full = [cl for cl in clusters if len(cl) == 3]
    assert len(full) == 2
    obs = [o for o in reconstruct_objects(cams, dets) if o.cls == "sports ball"]
    assert len(obs) == 2 and all(o.mode == "tri" for o in obs)
    for X in (X1, X2):
        assert min(np.linalg.norm(o.pos - X) for o in obs) < 0.02


def test_mono_floor_object():
    cam = make_cam([-2, -2, 2])
    X = np.array([0.3, 0.2, 0.1])  # мяч радиусом 10 см лежит на полу
    d = synth_detections(cam, [("sports ball", X, 0.1)])[0]
    o = mono_object("a", cam, d)
    assert o is not None and o.mode == "mono"
    assert np.linalg.norm(o.floor[:2] - X[:2]) < 0.15


def render_tag(cam: CameraModel, tag_id: int, center, side_m: float, ss: int = 3):
    """Метка ArUco лежит горизонтально (верх метки смотрит в +Y мира) — как её видит камера."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, TAG_DICT))
    n, margin = 280, 70  # сторона чёрного квадрата и белое поле, пиксели текстуры
    tex = cv2.copyMakeBorder(cv2.aruco.generateImageMarker(d, tag_id, n), margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)
    N = tex.shape[0]
    k = side_m / n  # метров на пиксель текстуры
    C = np.asarray(center, float)
    ex, ey = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    K = cam.K * np.array([[ss], [ss], [1]])
    size = (cam.size[0] * ss, cam.size[1] * ss)
    H = K @ np.column_stack([cam.R @ (k * ex), cam.R @ (-k * ey), cam.R @ (C - N / 2 * k * ex + N / 2 * k * ey) + cam.t])
    img = cv2.warpPerspective(tex, H, size, borderValue=170)
    img = cv2.resize(img, cam.size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_aruco_detect_and_triangulate():
    cams = three_cams()
    X = np.array([0.2, -0.1, 0.3])
    det = ArucoObjectDetector()
    dets = {}
    for c, m in cams.items():
        ds = det(render_tag(m, 7, X, 0.15))
        assert [d.cls for d in ds] == ["tag:7"], f"{c}: {[d.cls for d in ds]}"
        dets[c] = ds
    obs = reconstruct_objects(cams, dets)
    assert len(obs) == 1 and obs[0].mode == "tri" and obs[0].cls == "tag:7"
    assert np.linalg.norm(obs[0].pos - X) < 0.01, obs[0].pos
    # одна камера: положение по размеру метки (solvePnP); до метки ~3 м, она ~45 px — ошибка в основном по глубине
    o = mono_object("a", cams["a"], dets["a"][0], tag_size_m=0.15)
    assert o is not None and np.linalg.norm(o.pos - X) < 0.06, o.pos
    # метки доски (DICT_4X4_50) за объекты не принимаются
    d4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    img = cv2.copyMakeBorder(cv2.aruco.generateImageMarker(d4, 3, 200), 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)
    assert det(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)) == []


def test_tracker_classes_separate():
    """Объекты разных классов в одной точке — разные треки, ID не пересекаются с людьми."""
    cams = three_cams()
    tracker = ObjectTracker(min_hits=1)
    X = np.array([0.0, 0.0, 0.3])
    dets = {c: synth_detections(m, [("sports ball", X, 0.1), ("cup", X + [0.05, 0, 0], 0.05)]) for c, m in cams.items()}
    out = tracker.update(0.0, reconstruct_objects(cams, dets))
    assert sorted(o["cls"] for o in out) == ["cup", "sports ball"]
    assert len({o["id"] for o in out}) == 2
    tracker.reset()
    assert tracker.update(0.1, []) == []


def test_extension_in_demo(tmp_path):
    """Сервер в демо-режиме: REST /api/objects/ работает, в сообщении frame есть objects."""
    from fastapi.testclient import TestClient

    from peleng.config import load_config
    from peleng.server import create_app

    app = create_app(load_config(), demo=True, data_dir=tmp_path)
    with TestClient(app) as client:
        r = client.get("/api/objects/")
        assert r.status_code == 200 and r.json()["mode"] == "off"
        assert client.post("/api/objects/", json={"mode": "bad"}).status_code == 400
        r = client.post("/api/objects/", json={"mode": "aruco"})
        assert r.json()["mode"] == "aruco"
        got = None
        with client.websocket_connect("/ws/stream") as ws:
            for _ in range(200):
                m = json.loads(ws.receive_text())
                if m.get("type") == "frame" and "objects" in m and m["objects"]:
                    got = m
                    break
        assert got is not None, "нет объектов в frame"
        assert got["objects"][0]["cls"] == "tag:7" and got["objects"][0]["kind"] == "object"
        # yolo в демо — синтетические объекты, модель не грузится; фильтр классов работает
        r = client.post("/api/objects/", json={"mode": "yolo", "classes": "sports ball"})
        assert r.json()["classes"] == ["sports ball"]
        clss = set()
        with client.websocket_connect("/ws/stream") as ws:
            for _ in range(200):
                m = json.loads(ws.receive_text())
                if m.get("type") == "frame" and m.get("objects"):
                    clss = {o["cls"] for o in m["objects"]}
                    break
        assert clss == {"sports ball"}
        r = client.post("/api/objects/", json={"mode": "off"})
        assert r.json()["mode"] == "off"
