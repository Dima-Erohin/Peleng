"""Проверка геометрии, калибровки и конвейера на синтетических данных (без камер)."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peleng.calibration import Board, calibrate_intrinsics, solve_extrinsics  # noqa: E402
from peleng.demo import DemoDetector, DemoWorld  # noqa: E402
from peleng.geometry import CameraModel, approx_K, essential_between, look_at, symmetric_epipolar_px, triangulate_robust  # noqa: E402
from peleng.reconstruct import reconstruct  # noqa: E402
from peleng.sources import Frame  # noqa: E402
from peleng.tracking import Tracker  # noqa: E402

SIZE = (1280, 720)


def make_cam(center, target=(0, 0, 0.5), hfov=70.0):
    R, t = look_at(np.array(center, float), np.array(target, float))
    return CameraModel(approx_K(*SIZE, hfov), np.zeros(5), SIZE, R, t)


def test_triangulation_and_epipolar():
    a = make_cam([-2, -2, 2])
    b = make_cam([2, -1.5, 2.2])
    c = make_cam([0, 2.5, 2.0])
    X = np.array([0.3, -0.2, 1.1])
    pts = [m.project(X)[0] for m in (a, b, c)]
    Xr, err, used = triangulate_robust([a, b, c], pts)
    assert np.linalg.norm(Xr - X) < 1e-6 and err < 1e-3 and len(used) == 3
    # выброс в одном виде отбрасывается
    pts_bad = [pts[0], pts[1] + np.array([80.0, 0]), pts[2]]
    Xr, err, used = triangulate_robust([a, b, c], pts_bad)
    assert used == [0, 2] and np.linalg.norm(Xr - X) < 1e-6
    E = essential_between(a, b)
    d = symmetric_epipolar_px(E, a.undistort(pts[0][None]), b.undistort(pts[1][None]), a.focal, b.focal)
    assert d[0] < 1e-6


def render_board(board: Board, cam: CameraModel, pps=300, margin=60, ss=3):
    """Рисует доску, лежащую на полу, так, как её видит камера (с суперсэмплингом — как реальная оптика)."""
    if ss > 1:
        big = CameraModel(cam.K * np.array([[ss], [ss], [1]]), cam.dist, (cam.size[0] * ss, cam.size[1] * ss), cam.R, cam.t)
        img = render_board(board, big, pps, margin, ss=1)
        return cv2.resize(img, cam.size, interpolation=cv2.INTER_AREA)
    img = board.image(pps, margin)
    S = np.array([[board.square_m / pps, 0, -margin * board.square_m / pps], [0, board.square_m / pps, -margin * board.square_m / pps], [0, 0, 1]])
    R_cb = cam.R @ board.R_wb
    t_cb = cam.R @ board.t_wb + cam.t
    H = cam.K @ np.column_stack([R_cb[:, 0], R_cb[:, 1], t_cb]) @ S
    out = cv2.warpPerspective(img, H, cam.size, borderValue=200)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


A3 = {"square_m": 0.07, "marker_m": 0.052}


@pytest.mark.parametrize(
    "center,cfg,tol",
    [
        ([-1.2, -0.9, 1.3], {}, 0.015),
        ([1.4, 0.6, 1.1], {}, 0.015),
        ([0.0, -1.5, 0.9], {}, 0.015),
        ([-2.0, -1.5, 2.0], A3, 0.015),
        # ~4 м до доски A3 под острым углом: ошибка уже несколько сантиметров — так и в жизни
        ([2.5, 2.0, 2.3], A3, 0.05),
    ],
)
def test_extrinsics_from_board(center, cfg, tol):
    board = Board(cfg)
    true = make_cam(center, target=(0, 0, 0), hfov=70)
    img = render_board(board, true)
    det = board.detect(img)
    assert det is not None and len(det.ids) >= 6
    res = solve_extrinsics(board, true, [det])
    assert res.ok and res.reproj_px < 1.0
    est = CameraModel(true.K, true.dist, SIZE, res.R, res.t)
    # ошибка положения камеры относительно расстояния до доски
    assert np.linalg.norm(est.center - true.center) < tol * np.linalg.norm(true.center)


def test_refine_focal_from_single_view():
    board = Board({})
    true = make_cam([-1.0, -1.0, 1.2], target=(0, 0, 0), hfov=70)
    img = render_board(board, true)
    det = board.detect(img)
    guess = CameraModel(approx_K(*SIZE, 60), np.zeros(5), SIZE)  # ошиблись с углом обзора на 10°
    res = solve_extrinsics(board, guess, [det], refine_f=True)
    assert res.ok and res.K is not None
    assert abs(res.K[0, 0] - true.K[0, 0]) / true.K[0, 0] < 0.05
    est = CameraModel(res.K, guess.dist, SIZE, res.R, res.t)
    assert np.linalg.norm(est.center - true.center) < 0.05


def test_intrinsics_multi_view():
    board = Board({})
    rng = np.random.default_rng(1)
    true_K = approx_K(*SIZE, 66)
    dets = []
    for _ in range(15):
        ang = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0.15, 0.4)
        cam = CameraModel(true_K, np.zeros(5), SIZE)
        cam.R, cam.t = look_at(np.array([r * np.cos(ang), r * np.sin(ang), rng.uniform(0.6, 1.0)]), rng.normal(0, 0.05, 3))
        d = board.detect(render_board(board, cam))
        if d is not None:
            dets.append(d)
    K, dist, rms = calibrate_intrinsics(dets, SIZE)
    assert rms < 1.0
    assert abs(K[0, 0] - true_K[0, 0]) / true_K[0, 0] < 0.03


def test_demo_pipeline_accuracy():
    world = DemoWorld(n_cams=3, n_people=2)
    det = DemoDetector(world, noise_px=1.5, drop_prob=0.05)
    tracker = Tracker()
    t0 = world.t0
    errs = []
    ids = set()
    for k in range(60):
        t = t0 + k / 15
        batch = [(c, Frame(None, t, k)) for c in world.cams]
        res = det(batch)
        dets = dict(zip(world.cams, res))
        obs = reconstruct(world.cams, dets)
        out = tracker.update(t, obs)
        truth = world.people(t)
        if k > 5:
            assert len(out) == 2
            for p in out:
                ids.add(p["id"])
                hip_truth = [tr[1][[11, 12]].mean(axis=0) for tr in truth]
                errs.append(min(np.linalg.norm(np.array(p["pos"]) - h) for h in hip_truth))
    assert len(ids) == 2, "ID должны быть стабильными"
    assert np.median(errs) < 0.05, f"медианная ошибка {np.median(errs):.3f} м"


def test_mono_fallback():
    world = DemoWorld(n_cams=1, n_people=1)
    det = DemoDetector(world, noise_px=1.0, drop_prob=0.0)
    t = world.t0 + 1.0
    res = det([("demo1", Frame(None, t, 1))])
    obs = reconstruct(world.cams, {"demo1": res[0]})
    assert len(obs) == 1 and obs[0].mode == "mono"
    truth = world.people(t)[0][1][[15, 16]].mean(axis=0)
    assert np.linalg.norm(obs[0].floor[:2] - truth[:2]) < 0.15


def test_frame_lazy_jpeg_and_nearest():
    from peleng.sources import PushSource

    img = np.full((360, 640, 3), 128, np.uint8)
    jpeg = cv2.imencode(".jpg", img)[1]
    src = PushSource("p")
    src.push_jpeg(jpeg, 10.0)
    f = src.latest()
    assert f._image is None, "JPEG не должен распаковываться при приёме"
    assert f.image.shape == (360, 640, 3)
    for k, ts in enumerate((10.10, 10.20, 10.30)):
        src.push(img, ts)
    assert abs(src.nearest(10.22).ts - 10.20) < 1e-9


def test_capture_on_demand(tmp_path):
    """Пачка берётся, когда пришли кадры от всех телефонов; калибровке достаётся полное разрешение."""
    from peleng.pipeline import Pipeline
    from peleng.sources import CameraRegistry, PushSource

    registry = CameraRegistry(tmp_path / "calib")
    sent: dict[str, list] = {"a": [], "b": []}
    for cid in sent:
        src = PushSource(cid)
        src.info = {"on_demand": True, "vw": 1280, "vh": 720}
        src.sender = sent[cid].append
        registry.add(src)
    cfg = {"pipeline": {"track_width": 640, "capture_timeout_s": 0.3}}
    pl = Pipeline(cfg, registry, lambda batch: [[] for _ in batch], lambda m: None, tmp_path)
    pl.request_frames()
    assert sent["a"][-1] == {"type": "capture", "w": 640}
    registry.sources["a"].push(np.zeros((360, 640, 3), np.uint8), time.time())
    assert pl._waiting(), "кадр от b ещё не пришёл"
    registry.sources["b"].push(np.zeros((360, 640, 3), np.uint8), time.time())
    assert not pl._waiting()
    with pl.full_res(["b"]):
        pl.request_frames()
        assert sent["a"][-1]["w"] == 640 and sent["b"][-1]["w"] == 0
    msg = pl.step()
    assert msg is not None and set(msg["detections"]) == {"a", "b"}
