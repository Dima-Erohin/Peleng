"""Тесты офлайн-конвейера на синтетике (без видео): камеры по модели Unreal, известная траектория.

Запуск:  python -m pytest offline/tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from offline.detect_base import Detection  # noqa: E402
from offline.evaluate import evaluate  # noqa: E402
from offline.tracker3d import Tracker3D, TrackerConfig, triangulate_ls  # noqa: E402
from offline.ue_data import parse_num, read_ue_csv, ue_camera_model, ue_to_world, world_to_ue, write_ue_csv  # noqa: E402

FPS = 30.0
# камеры как в тестовом наборе (Unreal: см, градусы), смотрят примерно в центр сцены
CAMS_UE = [
    ((-24702.0, 6623.6, 1200.0), (0.0, -50.0, 0.0), 35.0),
    ((-45784.0, -6500.0, 7911.9), (-10.0, 5.0, 0.0), 60.0),
    ((12600.0, -22700.0, 1400.0), (-3.1, 115.1, -0.7), 35.0),
    ((33281.7, 13996.7, 7146.7), (-10.0, -160.0, 0.0), 50.0),
    ((9799.8, -3900.0, 1487.5), (-7.0, -164.0, -4.4), 20.0),
]


def make_cams():
    out = []
    for loc, (p, y, r), f in CAMS_UE:
        out.append(ue_camera_model(loc, p, y, r, f))
    return out


def look_at_ue(loc, target_ue, f):
    """Камера Unreal в loc, смотрящая на target (pitch/yaw из направления)."""
    d = np.asarray(target_ue, float) - np.asarray(loc, float)
    yaw = np.degrees(np.arctan2(d[1], d[0]))
    pitch = np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1])))
    return ue_camera_model(loc, pitch, yaw, 0.0, f)


def truth_traj(n=300):
    """Плавная траектория (мир, м): разгон + поворот, высота ~2 м, скорость ~15 м/с."""
    t = np.arange(n) / FPS
    x = -60 + 15 * t
    y = -40 + 20 * np.sin(t / 4.0)
    z = 2.0 + 0.3 * np.sin(t)
    return np.stack([x, y, z], 1)


def synth_dets(cams, X, vis, rng, size_m=4.0, noise_px=1.0, clutter=True):
    """Детекции: рамка вокруг проекции точки, размер по дальности; иногда ложные детекции."""
    dets = []
    for ci, cam in enumerate(cams):
        dl = []
        if vis[ci]:
            uv = cam.project(X)[0] + rng.normal(0, noise_px, 2)
            z = cam.depth(X)[0]
            s = cam.K[0, 0] * size_m / z
            if 0 <= uv[0] < 1920 and 0 <= uv[1] < 1080 and z > 0:
                dl.append(Detection((uv[0] - s, uv[1] - s / 2, uv[0] + s, uv[1] + s / 2), tuple(uv), 0.8, "truck", "synth"))
        if clutter and rng.random() < 0.3:
            u, v = rng.uniform(0, 1920), rng.uniform(0, 1080)
            dl.append(Detection((u - 20, v - 10, u + 20, v + 10), (u, v), 0.3, "motion", "synth"))
        dets.append(dl)
    return dets


def run_synth(n=300, vis_fn=None, seed=0, cfg=None):
    T = truth_traj(n)
    center = world_to_ue(T.mean(0))
    cams = [look_at_ue(loc, center, f) for loc, _, f in CAMS_UE]
    rng = np.random.default_rng(seed)
    frames = []
    for k in range(n):
        vis = vis_fn(k) if vis_fn else [True] * len(cams)
        frames.append(synth_dets(cams, T[k], vis, rng))
    return T, cams, frames


def track_frames(cams, frames, cfg=None):
    tr = Tracker3D(cfg or TrackerConfig(fps=FPS))
    out = []
    for k, d in enumerate(frames):
        r = tr.step(k, cams, d)
        out.append(np.full(3, np.nan) if r.pos is None else r.pos)
    return np.array(out), tr


# ---------------------------------------------------------------------------

def test_coordinate_roundtrip_and_projection():
    P = np.array([[123.0, -456.0, 789.0], [-21500.0, -12800.0, 300.0]])
    assert np.allclose(world_to_ue(ue_to_world(P)), P)
    cam = ue_camera_model((0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 35.0)
    f = 35.0 / 23.76 * 1920
    # точка в 100 м впереди, 10 м вправо (Unreal +Y), 5 м вверх
    uv = cam.project(ue_to_world([10000.0, 1000.0, 500.0]))[0]
    assert np.allclose(uv, [960 + f * 0.1, 540 - f * 0.05], atol=1e-6)
    assert abs(np.linalg.det(cam.R) - 1) < 1e-9


def test_csv_parser_robust(tmp_path):
    p = tmp_path / "t.csv"
    txt = ("Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;20\r\n"
           "0;1,5;2;3;0;0;0;;;матрица;23,76*13,365мм\r\n"
           "1;авг.13;2;3;0;0;0;;;;\r\n"
           "2;3.5;2;3;0;0;0;;;;\r\n"
           "2;3.5;2;3;0;0;0;;;;\r\n")
    p.write_bytes(txt.encode("cp1251"))
    t = read_ue_csv(p, ("Loc_X", "Loc_Y", "Loc_Z", "Rot_Roll", "Rot_Pitch", "Rot_Yaw"))
    assert t.n == 3 and t.meta["focal_mm"] == 20 and t.meta["sensor_mm"] == (23.76, 13.365)
    assert t.bad_frames == [1]
    assert np.allclose(t["Loc_X"], [1.5, 2.5, 3.5])
    assert np.isnan(parse_num("июл.25")) and parse_num("-1,25") == -1.25
    out = tmp_path / "o.csv"
    write_ue_csv(out, np.arange(2), np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]]))
    t2 = read_ue_csv(out, ("Loc_X", "Loc_Y", "Loc_Z"), interpolate=False)
    assert t2["Loc_X"][0] == 1.0 and np.isnan(t2["Loc_X"][1])


def test_triangulation_exact():
    T = truth_traj(5)
    center = world_to_ue(T.mean(0))
    cams = [look_at_ue(loc, center, f) for loc, _, f in CAMS_UE]
    X, cov, errs = triangulate_ls(cams[:3], [c.project(T[2])[0] for c in cams[:3]], [1.0] * 3)
    assert np.linalg.norm(X - T[2]) < 1e-3 and errs.max() < 1e-3


def test_multicam_accuracy():
    T, cams, frames = run_synth(300)
    E, _ = track_frames(cams, frames)
    r = evaluate(world_to_ue(E), world_to_ue(T))
    assert r["err3d"]["rmse"] < 1.5, r["err3d"]
    assert r["valid_frames"] == 300


def test_single_camera_interval():
    """Кадры 100–199 объект видит одна камера — EKF по лучу + дальность по размеру удерживают трек."""
    def vis(k):
        return [True] * 5 if (k < 100 or k >= 200) else [False, True, False, False, False]

    T, cams, frames = run_synth(300, vis)
    E, tr = track_frames(cams, frames)
    err = np.linalg.norm(E - T, axis=1)
    modes = [h.mode for h in tr.history]
    assert modes[150] == "mono"
    assert np.nanmax(err[100:200]) < 15.0, np.nanmax(err[100:200])
    assert np.sqrt(np.nanmean(err[200:] ** 2)) < 2.0
    # без измерения дальности по размеру хуже или так же, но тоже не разваливается
    E2, _ = track_frames(cams, frames, TrackerConfig(fps=FPS, size_range=False))
    err2 = np.linalg.norm(E2 - T, axis=1)
    assert np.nanmax(err2[100:200]) < 40.0


def test_causality_truncation():
    """Результат кадра k не меняется, если обрезать данные после k."""
    T, cams, frames = run_synth(200, lambda k: [True] * 5 if k % 50 < 30 else [False, True, False, False, False])
    full, _ = track_frames(cams, frames)
    for k in (40, 99, 150):
        part, _ = track_frames(cams, frames[: k + 1])
        assert np.allclose(part, full[: k + 1], equal_nan=True)


def test_rejects_far_clutter():
    """Ложные детекции далеко от объекта не утаскивают трек."""
    T, cams, frames = run_synth(200)
    rng = np.random.default_rng(5)
    for k in range(60, 140):
        for ci in range(5):
            u, v = rng.uniform(0, 1920), rng.uniform(0, 1080)
            frames[k][ci].append(Detection((u - 40, v - 20, u + 40, v + 20), (u, v), 0.9, "truck", "synth"))
    E, _ = track_frames(cams, frames)
    err = np.linalg.norm(E - T, axis=1)
    assert np.nanmax(err[10:]) < 3.0


@pytest.mark.parametrize("model", ["cv", "ca"])
def test_models(model):
    T, cams, frames = run_synth(150)
    E, _ = track_frames(cams, frames, TrackerConfig(fps=FPS, model=model))
    assert np.sqrt(np.nanmean(np.linalg.norm(E - T, axis=1) ** 2)) < 2.0
