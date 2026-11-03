"""Карта покрытия на синтетических камерах (без реальных камер)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peleng import accuracy as acc  # noqa: E402
from peleng import coverage as cov  # noqa: E402
from peleng.geometry import CameraModel, approx_K, look_at  # noqa: E402

SIZE = (1280, 720)


def make_cam(center, target=(0, 0, 0.9), hfov=70.0, size=SIZE):
    R, t = look_at(np.array(center, float), np.array(target, float))
    return CameraModel(approx_K(*size, hfov), np.zeros(5), size, R, t)


def at(cams, *pts, **params):
    return cov.evaluate_points(cams, np.array(pts, float), params)


def test_visibility_counts():
    cams = {"a": make_cam([-3, 0, 2]), "b": make_cam([0, -3, 2], target=(1, 0, 0.9))}
    ev = at(cams, [0, 0, 0.9], [-4.5, 0, 0.9], [-1.5, 0, 0.9])
    # центр — перед обеими камерами; за спиной камеры a — никто; у a сбоку, вне кадра b — одна
    assert ev["counts"].tolist() == [2, 0, 1]
    assert ev["visible"]["a"].tolist() == [True, False, True]
    assert np.isfinite(ev["sigma_h_cm"][0]) and np.isnan(ev["sigma_h_cm"][2])
    assert ev["angle_deg"][0] > 80
    # слишком близко к камере (глубина < 0.2 м) и слишком далеко — не видно
    one = {"a": cams["a"]}
    assert at(one, [-2.9, 0, 1.95])["counts"][0] == 0
    assert at(one, [0, 0, 0.9], max_range_m=2.0)["counts"][0] == 0
    # у самого края кадра — не видно из-за отступа margin_frac
    m = cams["a"]
    edge = m.center + (m.R.T @ np.array([(SIZE[0] / 2 - 5) / m.K[0, 0], 0, 1.0])) * 3.0
    assert at(one, edge, margin_frac=0.0)["counts"][0] == 1
    assert at(one, edge, margin_frac=0.03)["counts"][0] == 0


def test_quality_parallel_vs_perpendicular():
    X = [0, 0, 0.9]
    perp = {"a": make_cam([-3, 0, 2]), "b": make_cam([0, -3, 2])}
    para = {"a": make_cam([-3, 0, 2]), "b": make_cam([-3, 0.4, 2])}
    ep, eq = at(perp, X), at(para, X)
    assert ep["counts"][0] == eq["counts"][0] == 2
    assert eq["angle_deg"][0] < 10 < 80 < ep["angle_deg"][0]
    assert eq["sigma_h_cm"][0] > 3 * ep["sigma_h_cm"][0]
    assert eq["quality"][0] < ep["quality"][0]
    # σ — та же, что у accuracy.pos_std (без дисторсии якобианы совпадают)
    s = acc.pos_std(list(perp.values()), np.array(X), 2.0)
    assert abs(ep["sigma_h_cm"][0] - np.hypot(s[0], s[1]) * 100) < 0.01


def test_uncalibrated_ignored_and_map():
    a, b = make_cam([-2.4, -1.9, 2.2]), make_cam([2.4, -1.9, 2.2])
    raw = CameraModel(approx_K(*SIZE, 70.0), np.zeros(5), SIZE)  # без положения
    room = {"size_m": [5, 4, 2.7], "board_offset_m": [0, 0]}
    r = cov.coverage_map({"a": a, "b": b, "raw": raw}, room, {"cell_m": 0.25})
    assert r["ok"] and [c["id"] for c in r["cameras"]] == ["a", "b"]
    assert r["bounds_from"] == "room" and r["nx"] == 20 and r["ny"] == 16 and r["origin"] == [-2.5, -2.0]
    assert len(r["counts"]) == len(r["quality"]) == r["nx"] * r["ny"] and max(r["counts"]) == 2
    s = r["summary"]
    assert abs(s["area_2plus_m2"] + s["area_1_m2"] + s["area_0_m2"] - 20.0) < 1e-6 and s["area_2plus_m2"] > 5
    # клетка в центре видна обеими, под камерой у стены — никем
    idx = lambda x, y: int((y - r["origin"][1]) // r["cell_m"]) * r["nx"] + int((x - r["origin"][0]) // r["cell_m"])  # noqa: E731
    assert r["counts"][idx(0.1, 0.1)] == 2 and r["quality"][idx(0.1, 0.1)] > 0
    assert r["counts"][idx(-2.3, -1.85)] == 0 and r["quality"][idx(-2.3, -1.85)] is None
    # некалиброванная камера одна — карта пустая, с понятным сообщением
    r0 = cov.coverage_map({"raw": raw}, None)
    assert not r0["ok"] and max(r0["counts"]) == 0 and r0["warnings"]


def test_bounds_from_cameras_and_limits():
    cams = {"a": make_cam([-3, 0, 2]), "b": make_cam([0, -3, 2])}
    r = cov.coverage_map(cams, {"size_m": None}, {"cell_m": 0.01, "max_range_m": 30})
    assert r["bounds_from"] == "cameras"
    assert r["nx"] * r["cell_m"] <= 12.0 + r["cell_m"] and r["ny"] * r["cell_m"] <= 12.0 + r["cell_m"]
    assert r["nx"] * r["ny"] <= 40000 and r["cell_m"] >= cov.MIN_CELL_M
    # модель под другое разрешение даёт ту же видимость
    small = {c: m.resized((640, 360)) for c, m in cams.items()}
    r2 = cov.coverage_map(small, {"size_m": None}, {"cell_m": 0.2})
    r1 = cov.coverage_map(cams, {"size_m": None}, {"cell_m": 0.2})
    assert r1["counts"] == r2["counts"]


def test_demo_server_coverage(tmp_path):
    from fastapi.testclient import TestClient

    from peleng.config import load_config
    from peleng.server import create_app

    cfg = load_config()
    cfg["demo"] = {"cameras": 3, "people": 1}
    app = create_app(cfg, demo=True, data_dir=tmp_path)
    with TestClient(app) as client:
        r = client.get("/api/coverage", params={"cell_m": 0.25, "height_m": 0.9}).json()
        assert r["ok"] and len(r["cameras"]) == 3 and r["bounds_from"] == "room"
        assert r["nx"] == 20 and r["ny"] == 16 and r["cell_m"] == 0.25 and r["height_m"] == 0.9
        assert r["summary"]["area_2plus_m2"] > 10 and r["summary"]["sigma_h_cm_median"] < 5
        assert client.get("/api/coverage", params={"cell_m": -1}).status_code == 400
        assert client.get("/api/coverage").json()["cell_m"] == 0.2
