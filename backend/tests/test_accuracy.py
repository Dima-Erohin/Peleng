"""Точность: σ триангуляции, проверка по доске, рулетка, хук pos_std_m (синтетика, без камер)."""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi import FastAPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peleng import accuracy as acc  # noqa: E402
from peleng.calibration import Board, BoardDetection  # noqa: E402
from peleng.ext import ExtContext  # noqa: E402
from peleng.ext import accuracy as acc_ext  # noqa: E402
from peleng.pipeline import FrameContext  # noqa: E402
from peleng.sources import CameraRegistry  # noqa: E402
from test_core import make_cam, render_board  # noqa: E402

X0 = np.array([0.0, 0.0, 0.9])


def test_sigma_decreases_with_third_camera():
    a = make_cam([-2.5, -2.0, 2.2], target=X0)
    b = make_cam([2.5, -2.0, 2.2], target=X0)
    c = make_cam([0.0, 2.8, 2.2], target=X0)
    s2 = acc.pos_std([a, b], X0)
    s3 = acc.pos_std([a, b, c], X0)
    assert s2 is not None and s3 is not None
    assert np.linalg.norm(s3) < np.linalg.norm(s2)
    assert (s3 <= s2 + 1e-12).all()
    # порядок величины: 2 px на 1280 px при 3 м — около сантиметра
    assert 0.002 < np.linalg.norm(s2) < 0.05
    assert acc.pos_std([a], X0) is None


def test_sigma_grows_when_angle_narrows():
    """Две камеры на расстоянии 3 м от точки; угол между ними сужается → σ растёт (в основном по глубине)."""
    prev = 0.0
    for ang in (90, 45, 20, 8):
        h = np.radians(ang / 2)
        a = make_cam(X0 + 3 * np.array([np.sin(h), -np.cos(h), 0.0]), target=X0)
        b = make_cam(X0 + 3 * np.array([-np.sin(h), -np.cos(h), 0.0]), target=X0)
        s = np.linalg.norm(acc.pos_std([a, b], X0))
        assert s > prev
        prev = s
        assert abs(acc.effective_angle_deg(acc.ray_angle_deg(a, b, X0)) - ang) < 3
    # камеры напротив друг друга — так же плохо, как рядом
    assert acc.effective_angle_deg(175.0) == 5.0


def test_covariance_matches_monte_carlo():
    """σ из якобиана совпадает с разбросом триангуляции при шуме σ px."""
    from peleng.geometry import triangulate_robust

    cams = [make_cam([-2.5, -2.0, 2.2], target=X0), make_cam([2.0, -2.5, 2.0], target=X0)]
    rng = np.random.default_rng(0)
    px = [c.project(X0)[0] for c in cams]
    Xs = [triangulate_robust(cams, [p + rng.normal(0, 2.0, 2) for p in px], reproj_thresh_px=100)[0] for _ in range(400)]
    emp = np.std(np.array(Xs), axis=0)
    est = acc.pos_std(cams, X0, 2.0)
    assert np.allclose(emp, est, rtol=0.2, atol=5e-4)


# ---------------------------------------------------------------------------
# проверка по доске
# ---------------------------------------------------------------------------

def _board_world(board: Board) -> np.ndarray:
    return board.board_to_world(board.all_corners)


def _projected_dets(board, cams, noise_px=0.0, seed=0):
    rng = np.random.default_rng(seed)
    Xw = _board_world(board)
    ids = np.arange(len(Xw))
    out = {}
    for cid, m in cams.items():
        px = m.project(Xw) + rng.normal(0, noise_px, (len(Xw), 2)) if noise_px else m.project(Xw)
        out[cid] = BoardDetection(px, ids, board.all_corners[ids])
    return out


BOARD_CAMS = {"a": [-1.2, -0.9, 1.3], "b": [1.4, -0.6, 1.1], "c": [0.2, 1.4, 1.2]}


def test_board_check_ideal_projection():
    board = Board({})
    cams = {k: make_cam(v, target=(0, 0, 0)) for k, v in BOARD_CAMS.items()}
    res = acc.board_check(board, cams, _projected_dets(board, cams))
    assert res["ok"], res
    assert res["corners"] == len(board.all_corners)
    assert res["max_err_mm"] < 0.01 and res["plane_max_mm"] < 0.01
    assert res["tilt_deg"] < 0.1 and res["span_err_mm"] < 0.01


def test_board_check_rendered_images():
    """Доска нарисована в кадры (как видит камера) → найдена детектором → ошибка ≈ 0 (доли мм)."""
    board = Board({})
    cams = {k: make_cam(v, target=(0, 0, 0)) for k, v in BOARD_CAMS.items()}
    dets = {k: acc.merge_detections([board.detect(render_board(board, m))], board) for k, m in cams.items()}
    assert all(d is not None for d in dets.values())
    res = acc.board_check(board, cams, dets)
    assert res["ok"], res
    assert res["median_err_mm"] < 1.0 and res["plane_rms_mm"] < 1.5, res


def test_board_check_with_noise_matches_expected():
    board = Board({})
    cams = {k: make_cam(v, target=(0, 0, 0)) for k, v in BOARD_CAMS.items()}
    rms = []
    for seed in range(5):
        res = acc.board_check(board, cams, _projected_dets(board, cams, noise_px=1.0, seed=seed), sigma_px=1.0)
        assert res["ok"]
        rms.append(res["rms_err_mm"])
    expected = res["expected_err_mm"]
    # ошибка расстояния между соседями того же порядка, что и ожидаемая из геометрии камер
    assert 0.4 * expected < np.mean(rms) < 2.5 * expected, (rms, expected)


def test_board_check_needs_two_cameras():
    board = Board({})
    cams = {"a": make_cam(BOARD_CAMS["a"], target=(0, 0, 0))}
    res = acc.board_check(board, cams, _projected_dets(board, cams))
    assert not res["ok"] and "≥ 2" in res["message"]


def test_merge_detections_median():
    board = Board({})
    ids = np.array([0, 1, 2])
    d1 = BoardDetection(np.array([[10.0, 10], [20, 20], [30, 30]]), ids, board.all_corners[ids])
    d2 = BoardDetection(np.array([[12.0, 10], [20, 22]]), ids[:2], board.all_corners[ids[:2]])
    d3 = BoardDetection(np.array([[11.0, 10]]), ids[:1], board.all_corners[ids[:1]])
    m = acc.merge_detections([d1, d2, d3], board)
    assert list(m.ids) == [0, 1, 2]
    assert np.allclose(m.corners[0], [11, 10]) and np.allclose(m.corners[2], [30, 30])


# ---------------------------------------------------------------------------
# рулетка
# ---------------------------------------------------------------------------

def test_tape_a_to_b():
    rng = np.random.default_rng(1)

    def samples(p):
        p = np.array(p, float)
        return [{"pos": (p + [0, 0, 0.9] + rng.normal(0, 0.01, 3)).tolist(), "floor": (p + rng.normal(0, 0.01, 3) * [1, 1, 0]).tolist(),
                 "std": [0.01, 0.012, 0.02], "mode": "tri"} for _ in range(15)]

    a = acc.average_point(samples([0.0, 0.0, 0.0]))
    b = acc.average_point(samples([2.4, 1.8, 0.0]))  # 3.0 м
    assert a["n"] == 15 and a["spread_m"] < 0.03 and a["tri_share"] == 1.0
    r = acc.tape_result(a, b, 3.0)
    assert r["horizontal"] and abs(r["system_m"] - 3.0) < 0.02
    assert abs(r["err_cm"]) < 2 and r["err_pct"] < 0.7
    assert 0.5 < r["expected_cm"] < 3
    # рулетка показала 3.10 → система «короче» на ~10 см
    r = acc.tape_result(a, b, 3.10)
    assert abs(r["err_cm"] + 10) < 2 and abs(r["err_pct"] - 3.2) < 0.7
    # объект без floor — 3D-расстояние
    a2 = {"pos": [0, 0, 0]}
    b2 = {"pos": [0, 0, 1.0]}
    r = acc.tape_result(a2, b2, 1.0)
    assert not r["horizontal"] and r["err_cm"] == 0.0


def test_history_and_summary(tmp_path):
    p = tmp_path / "accuracy.json"
    assert acc.load_history(p) == []
    acc.append_history(p, {"type": "tape", "err_cm": -3.0, "err_pct": 1.0})
    acc.append_history(p, {"type": "tape", "err_cm": 5.0, "err_pct": 2.0})
    acc.append_history(p, {"type": "board", "ok": True, "median_err_mm": 1.5, "max_err_mm": 3.0, "plane_rms_mm": 1.0, "cams": ["a", "b"], "corners": 12})
    s = acc.summarize_history(acc.load_history(p))
    assert s["tape_checks"] == 2 and s["tape_median_err_cm"] == 4.0 and s["tape_max_err_cm"] == 5.0
    assert s["board_checks"] == 1 and s["board_last"]["median_err_mm"] == 1.5


def test_camera_report_warns_on_collinear_cameras():
    a = make_cam([-3.0, 0.0, 1.0], target=X0)
    b = make_cam([3.0, 0.1, 1.0], target=X0)  # напротив, почти на одной линии с центром
    c = make_cam([0.0, -3.0, 2.0], target=X0)
    for m, r in ((a, 0.4), (b, 0.5), (c, 2.1)):
        m.meta.update(intrinsics="board", intrinsics_rms=0.3, extrinsics_reproj_px=r)
    rep = acc.camera_report({"a": a, "b": b, "c": c}, X0)
    ab = next(p for p in rep["pairs"] if (p["a"], p["b"]) == ("a", "b"))
    assert not ab["ok"] and ab["effective_deg"] < 20
    assert all(p["ok"] for p in rep["pairs"] if p is not ab)
    assert any("a–b" in w for w in rep["warnings"])
    assert any("c:" in w and "1.5" in w for w in rep["warnings"])
    assert rep["zone_sigma_cm"] and all(c["sees_zone_center"] for c in rep["cameras"])


# ---------------------------------------------------------------------------
# хук конвейера
# ---------------------------------------------------------------------------

def test_hook_adds_pos_std(tmp_path):
    cams = {"a": make_cam([-2.5, -2.0, 2.2], target=X0), "b": make_cam([2.5, -2.0, 2.2], target=X0), "c": make_cam([0, 2.8, 2.2], target=X0)}
    registry = CameraRegistry(tmp_path / "calib")
    registry.set_model("c", cams["c"], save=False)  # «c» нет в ctx.cams — модель берётся из реестра под размер кадра
    pipeline = SimpleNamespace(frame_hooks=[], reset_hooks=[])
    app = FastAPI()
    ext_ctx = ExtContext({}, registry, pipeline, None, Board({}), {}, tmp_path)
    acc_ext.register(app, ext_ctx)
    assert len(pipeline.frame_hooks) == 1 and acc_ext.ORDER > 50
    msg = {
        "type": "frame",
        "persons": [
            {"id": 1, "pos": X0.tolist(), "floor": [0, 0, 0], "mode": "tri", "cams": ["a", "b"]},
            {"id": 2, "pos": [1.0, 0.5, 0.9], "floor": [1.0, 0.5, 0], "mode": "mono", "cams": ["a"]},
            {"id": 3, "pos": X0.tolist(), "floor": [0, 0, 0], "mode": "tri", "cams": ["a", "b", "c"]},
        ],
        "objects": [{"id": 101, "kind": "object", "cls": "tag:7", "pos": [0.3, 0.2, 0.5], "mode": "tri", "cams": ["b", "c"]}],
        "frame_size": {"a": list(cams["a"].size), "b": list(cams["b"].size), "c": list(cams["c"].size)},
    }
    fctx = FrameContext([], {"a": cams["a"], "b": cams["b"]}, {}, 100.0, msg)
    pipeline.frame_hooks[0](fctx)
    p1, p2, p3 = msg["persons"]
    assert len(p1["pos_std_m"]) == 3 and all(v > 0 for v in p1["pos_std_m"])
    assert "pos_std_m" not in p2
    assert np.linalg.norm(p3["pos_std_m"]) < np.linalg.norm(p1["pos_std_m"])
    assert np.allclose(p1["pos_std_m"], acc.pos_std([cams["a"], cams["b"]], X0), atol=1e-4)
    assert "pos_std_m" in msg["objects"][0]


def test_demo_server_frames_and_report(tmp_path):
    """Сервер в демо-режиме: в frame есть pos_std_m, отчёт и точки A/B/рулетка работают."""
    from fastapi.testclient import TestClient

    from peleng.config import load_config
    from peleng.server import create_app

    cfg = load_config()
    n_cams = 3
    cfg["demo"] = {"cameras": n_cams, "people": 2}
    cfg["accuracy"] = {"avg_s": 0.3, "board_collect_s": 0.3}
    app = create_app(cfg, demo=True, data_dir=tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            got = None
            for _ in range(200):
                m = ws.receive_json()
                if m["type"] == "frame" and any(p.get("mode") == "tri" and not p.get("predicted") for p in m["persons"]):
                    got = m
                    break
        assert got is not None, "нет кадров с людьми"
        tri = [p for p in got["persons"] if p["mode"] == "tri"]
        assert all("pos_std_m" in p for p in tri)
        assert all(0 < np.linalg.norm(p["pos_std_m"]) < 0.2 for p in tri)
        rep = client.get("/api/accuracy/report").json()
        assert len(rep["cameras"]) == n_cams and len(rep["pairs"]) == 3 and rep["zone_sigma_cm"]
        ra = client.post("/api/accuracy/point/a", json={}).json()
        assert ra["n"] > 0 and "pos" in ra
        rb = client.post("/api/accuracy/point/b", json={}).json()
        assert rb["key"] == ra["key"]
        rt = client.post("/api/accuracy/tape", json={"measured_m": 1.0}).json()
        assert rt["type"] == "tape" and "err_cm" in rt
        assert len(client.get("/api/accuracy/history").json()) == 1
        # в демо доски в кадрах нет — проверка честно сообщает об этом
        rbd = client.post("/api/accuracy/board", json={}).json()
        assert rbd["ok"] is False
        app.state.pipeline.stop()  # чтобы конвейер не публиковал в закрытый цикл событий при выходе
        time.sleep(0.3)
