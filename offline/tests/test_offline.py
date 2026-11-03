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


def box_dets(cams, T, k, vis, rng, dims=(8.0, 2.5, 3.5), noise_px=1.0):
    """Рамки — описанные прямоугольники проекции параллелепипеда, ориентированного по скорости."""
    v = T[min(k + 1, len(T) - 1)] - T[max(k - 1, 0)]
    f = np.array([v[0], v[1], 0.0])
    f /= np.linalg.norm(f)
    lft = np.array([-f[1], f[0], 0.0])
    L, W, H = dims
    sg = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], dtype=float) * 0.5
    C = T[k] + sg[:, :1] * L * f + sg[:, 1:2] * W * lft + sg[:, 2:3] * H * np.array([0, 0, 1.0])
    out = []
    for ci, cam in enumerate(cams):
        if not vis[ci]:
            out.append([])
            continue
        uv = cam.project(C)
        b = np.r_[uv.min(0), uv.max(0)] + rng.normal(0, noise_px, 4)
        out.append([Detection(tuple(b), ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), 0.8, "truck", "synth")])
    return out, T[k] + f * L / 2


def test_box_model_front_anchor_and_mono():
    """Модель «параллелепипед»: размеры оцениваются, выход «перёд» точен; интервал одной камеры держится."""
    n = 300
    T = truth_traj(n)
    center = world_to_ue(T.mean(0))
    cams = [look_at_ue(loc, center, f) for loc, _, f in CAMS_UE]
    rng = np.random.default_rng(1)
    frames, fronts = [], []
    for k in range(n):
        vis = [True] * 5 if (k < 150 or k >= 230) else [False, True, False, False, False]
        d, fr = box_dets(cams, T, k, vis, rng)
        frames.append(d)
        fronts.append(fr)
    fronts = np.array(fronts)
    E, tr = track_frames(cams, frames, TrackerConfig(fps=FPS, meas="box", anchor="front"))
    err = np.linalg.norm(E - fronts, axis=1)
    assert np.sqrt(np.mean(err[60:150] ** 2)) < 1.0, err[60:150].max()
    assert np.nanmax(err[150:230]) < 6.0, np.nanmax(err[150:230])
    L = tr.x[tr.nk]
    assert abs(L - 8.0) < 1.0, tr.x[tr.nk:]


# ---------------------------------------------------------------------------
# Синтетика «дрон»: подъём, зависание, пролёт; реальные камеры тестового набора, частичная видимость
# ---------------------------------------------------------------------------

def drone_traj(n=750):
    """Плавная 3D-траектория (мир, м) около камеры 5: подъём 0–150, зависание 150–300, пролёт 300–600,
    торможение и зависание 600–750."""
    p = np.zeros((n, 3))
    x = np.array([-60.0, 30.0, 8.0])  # рабочая система: (X_ue, -Y_ue, Z_ue)/100
    v = np.zeros(3)
    for k in range(n):
        t = k / FPS
        if k < 150:
            vt = np.array([0.0, 0.0, 2.0])
        elif k < 300:
            vt = np.array([0.2 * np.sin(t), 0.2 * np.cos(t), 0.0])
        elif k < 600:
            a = (k - 300) / 300 * np.pi / 2
            vt = 12.0 * np.array([np.cos(a), -np.sin(a), 0.05])
        else:
            vt = np.zeros(3)
        v += np.clip(vt - v, -4.0 / FPS, 4.0 / FPS)  # ускорение не больше 4 м/с²
        x = x + v / FPS
        p[k] = x
    return p


def drone_vis(k):
    """Видимость по камерам (1..5): дрон много времени виден одной камерой, есть разрыв без камер."""
    v = [False] * 5
    if k < 566:
        v[4] = True
    if 200 <= k < 450:
        v[0] = v[1] = True
    if 400 <= k < 520:
        v[2] = True
    if 640 <= k:
        v[2] = True
    return v


def run_drone(cfg, seed=0, size_m=0.5, noise_px=1.0):
    T = drone_traj()
    center = world_to_ue(T.mean(0))
    cams = [look_at_ue(loc, center, f) for loc, _, f in CAMS_UE]
    rng = np.random.default_rng(seed)
    frames = []
    for k in range(len(T)):
        vis = drone_vis(k)
        dl = []
        for ci, cam in enumerate(cams):
            d = []
            if vis[ci]:
                uv = cam.project(T[k])[0] + rng.normal(0, noise_px, 2)
                s = max(2.0, cam.K[0, 0] * size_m / cam.depth(T[k])[0]) / 2
                d.append(Detection((uv[0] - s, uv[1] - s, uv[0] + s, uv[1] + s), tuple(uv), 0.7, "drone", "synth"))
            if rng.random() < 0.2:  # ложные срабатывания
                u, w = rng.uniform(0, 1920), rng.uniform(0, 1080)
                d.append(Detection((u - 5, w - 5, u + 5, w + 5), (u, w), 0.4, "drone", "synth"))
            dl.append(d)
        frames.append(dl)
    E, tr = track_frames(cams, frames, cfg)
    return T, E, tr


def test_drone_synthetic():
    """Дрон: ошибка мала там, где >= 2 камер; с априорным размером держится и по одной камере; все кадры с оценкой."""
    cfg = TrackerConfig(fps=FPS, object_size_m=0.5)
    T, E, tr = run_drone(cfg)
    err = np.linalg.norm(E - T, axis=1)
    modes = np.array([h.mode for h in tr.history])
    ncam = np.array([h.n_cams for h in tr.history])
    assert np.isfinite(E).all()  # трек создан с первого кадра (по одной камере и размеру)
    assert np.sqrt(np.mean(err[(ncam >= 2) & (np.arange(len(T)) > 230)] ** 2)) < 1.0
    assert np.median(err[100:200]) < 15.0  # одна камера, глубина по размеру (σ 50 %) + движение
    assert (modes[566:640] == "predict").all()  # без камер — прогноз
    assert np.max(err[566:640]) < 20.0
    assert np.sqrt(np.mean(err[680:] ** 2)) < 15.0


def test_static_single_camera_size_prior():
    """Неподвижный объект виден одной камерой (как дрон на тестовых видео): луч + дальность по априорной ширине.
    При верном размере ошибка мала, кадры помечены mono_size_prior; после пропадания — прогноз на месте."""
    cam = ue_camera_model((9799.8, -3900.0, 1487.5), -7.0, -164.0, -4.4, 20.0)
    X = cam.center + 11.0 * cam.R[2] + 2.0 * cam.R[0]  # 11 м перед камерой
    rng = np.random.default_rng(3)
    frames = []
    for k in range(400):
        d = []
        if k < 300:
            uv = cam.project(X)[0] + rng.normal(0, 0.5, 2)
            w = cam.K[0, 0] * 0.5 / cam.depth(X)[0]
            d.append(Detection((uv[0] - w / 2, uv[1] - w / 4, uv[0] + w / 2, uv[1] + w / 4), tuple(uv), 0.99, "drone", "synth"))
        frames.append([d])
    cfg = TrackerConfig(fps=FPS, object_size_m=0.5, size_dim="width", accel_std=1.0, point="detector")
    E, tr = track_frames([cam], frames, cfg)
    err = np.linalg.norm(E - X, axis=1)
    modes = [h.mode for h in tr.history]
    assert modes[100] == "mono_size_prior" and modes[350] == "predict"
    assert np.median(err[50:300]) < 0.5 and err[399] < 3.0


def test_moving_cameras_tracker():
    """Камеры движутся (положение и поворот по кадрам, как в CSV): модель камеры на каждый кадр — RMSE в норме."""
    n = 300
    T = truth_traj(n)
    rng = np.random.default_rng(7)
    base = [np.array(loc, float) for loc, _, _ in CAMS_UE[:4]]
    focal = [f for _, _, f in CAMS_UE[:4]]
    frames, cams_k = [], []
    for k in range(n):
        cams = []
        for i, (b, f) in enumerate(zip(base, focal)):
            # камера едет по кругу радиусом 30 м и поднимается, всё время смотрит на объект (с запаздыванием)
            a = k / FPS * 0.3 + i
            loc = b + np.array([3000 * np.cos(a), 3000 * np.sin(a), 20.0 * k])
            target = world_to_ue(T[max(k - 15, 0)])
            cams.append(look_at_ue(loc, target, f))
        cams_k.append(cams)
        dl = []
        for ci, cam in enumerate(cams):
            uv = cam.project(T[k])[0] + rng.normal(0, 1.0, 2)
            s = cam.K[0, 0] * 4.0 / cam.depth(T[k])[0]
            dl.append([Detection((uv[0] - s, uv[1] - s / 2, uv[0] + s, uv[1] + s / 2), tuple(uv), 0.8, "truck", "synth")]
                      if (ci != 2 or k < 150) else [])
        frames.append(dl)
    tr = Tracker3D(TrackerConfig(fps=FPS))
    E = np.array([(lambda r: np.full(3, np.nan) if r.pos is None else r.pos)(tr.step(k, cams_k[k], frames[k])) for k in range(n)])
    err = np.linalg.norm(E - T, axis=1)
    assert np.sqrt(np.nanmean(err[10:] ** 2)) < 1.5


def test_background_detector_moving_camera_off():
    """Вычитание фона при перемещении камеры (не чистый поворот) сбрасывается и не даёт ложных пятен."""
    from offline.basic_detectors import BackgroundDetector

    rng = np.random.default_rng(0)
    img = (rng.random((1080, 1920, 3)) * 255).astype(np.uint8)
    det = BackgroundDetector(warmup=2)
    out = []
    for k in range(20):
        cam = ue_camera_model((100.0 * k, 0.0, 500.0), 0.0, 0.0, 0.0, 35.0)  # камера едет
        shifted = np.roll(img, 5 * k, axis=1)
        out.append(det.detect(1, k, shifted, cam))
    assert all(len(o) == 0 for o in out)


def test_run_robust_inputs(tmp_path):
    """Произвольный набор: 3 камеры (у одной нет видео), битые ячейки, без эталона, видео короче CSV,
    объект с недоступным детектором — прогон не падает, CSV на все кадры."""
    import cv2
    import json as _json

    from offline.run import main

    n_csv, n_vid = 40, 30
    for i, (loc, rot, f) in enumerate(CAMS_UE[:3], start=1):
        lines = ["Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;" + str(f)]
        for k in range(n_csv):
            x = "авг.13" if (i == 2 and k == 5) else f"{loc[0] + 10 * k:.2f}".replace(".", ",")
            lines.append(f"{k};{x};{loc[1]};{loc[2]};{rot[2]};{rot[0]};{rot[1]};;;" + ("матрица;23,76*13,365мм" if k == 0 else ";"))
        (tmp_path / f"object_transform_Cam{i}.csv").write_bytes("\r\n".join(lines).encode("cp1251"))
        if i < 3:
            vw = cv2.VideoWriter(str(tmp_path / f"V.Cam{i}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 180))
            for k in range(n_vid):
                im = np.full((180, 320, 3), 120, np.uint8)
                cv2.rectangle(im, (10 + 3 * k, 80), (40 + 3 * k, 100), (255, 255, 255), -1)
                vw.write(im)
            vw.release()
    objs = [{"name": "obj", "out_csv": "obj.csv", "detector": "bg", "tracker": {"point": "center"}},
            {"name": "ghost", "out_csv": "ghost.csv", "detector": "no.such.module:Nope"}]
    (tmp_path / "objs.json").write_text(_json.dumps(objs))
    out = tmp_path / "out"
    r = main(["--data", str(tmp_path), "--out", str(out), "--objects-file", str(tmp_path / "objs.json"), "--no-plots"])
    assert (out / "obj.csv").exists() and not (out / "ghost.csv").exists()
    rows = (out / "obj.csv").read_text().strip().splitlines()
    assert rows[0] == "Frame;Loc_X;Loc_Y;Loc_Z" and len(rows) == 1 + n_vid
    assert (out / "viewer.json").exists()


def _arc_est(R=100.0, n=240, step_m=0.5, y_pivot=5.0, noise=0.0, seed=0):
    """Дуга окружности радиуса R (центр в (0, R) по (x, z) three), курс +x, центр дуги слева направо — к +z."""
    rng = np.random.default_rng(seed)
    th = np.arange(n) * step_m / R
    P3 = np.stack([R * np.sin(th), np.full(n, y_pivot), R - R * np.cos(th)], 1) + rng.normal(0, noise, (n, 3))
    return np.stack([P3[:, 0], P3[:, 2], P3[:, 1]], 1) * 100.0  # three -> UE (см)


def test_road_arc_geometry():
    """Синтетическая дуга: ось на окружности, кромки на ±width/2, высота = pivot − h_pivot, выброс отброшен."""
    from offline.road import build_road

    R, w, h = 100.0, 4.0, 3.1
    est = _arc_est(R)
    est[100] += [3000.0, 0.0, 0.0]  # выброс 30 м
    road = build_road(est, ["tri"] * len(est), 30.0, width_m=w, h_pivot_m=h, smooth_m=6.0)
    C, L, Rr = (np.array(road[k]) for k in ("centerline", "left", "right"))
    assert road["units"] == "three_m" and road["width_m"] == w and road["source"] == "trajectory_smoothed"
    rad = lambda A: np.hypot(A[:, 0], A[:, 2] - R)
    assert np.all(np.abs(rad(C) - R) < 0.2)
    # курс +x, центр дуги на +z: слева по ходу — внешняя сторона (радиус больше)
    assert np.all(np.abs(rad(L) - (R + w / 2)) < 0.25) and np.all(np.abs(rad(Rr) - (R - w / 2)) < 0.25)
    assert np.allclose(np.hypot(*(L - Rr)[:, [0, 2]].T), w, atol=2e-3)
    assert np.allclose(C[:, 1], 5.0 - h, atol=1e-6) and np.allclose(L[:, 1], C[:, 1]) and np.allclose(Rr[:, 1], C[:, 1])
    steps = np.hypot(*np.diff(C[:, [0, 2]], axis=0).T)
    assert abs(np.median(steps) - 2.0) < 0.05 and road["uncertain"] == []
    assert abs(road["length_m"] - 120.0) < 2.0
    assert np.allclose(np.array(road["ue_cm"]["centerline"])[:, 2] / 100.0, C[:, 1], atol=0.01)  # копия в Unreal


def test_road_uncertain_and_none():
    """`none` отбрасывается, длинный predict помечается uncertain, короткий — нет; ширина по умолчанию 1.6 × грузовик."""
    from offline.road import build_road
    from offline.presets import TRUCK_WIDTH_M

    est = _arc_est(n=300, step_m=0.5)
    modes = ["none"] * 20 + ["tri"] * 100 + ["predict"] * 60 + ["tri"] * 60 + ["predict"] * 10 + ["tri"] * 50
    road = build_road(est, modes, 30.0)
    assert abs(road["width_m"] - 1.6 * TRUCK_WIDTH_M) < 1e-9
    assert abs(road["length_m"] - (300 - 20) * 0.5) < 2.0  # первые 20 кадров не вошли
    assert len(road["uncertain"]) == 1  # 60 кадров predict (2 с) — да, 10 кадров — нет
    i0, i1 = road["uncertain"][0]
    assert 0 <= i0 <= i1 < road["n_points"] and 0.35 < (i0 + i1) / 2 / road["n_points"] < 0.55
    assert build_road(est[:1], ["tri"], 30.0) is None


def test_road_in_viewer_and_disable():
    from offline.road import add_road

    est = _arc_est(n=100)
    v = {"fps": 30.0, "objects": [{"name": "truck", "estimate": {"ue_cm": est.tolist(), "mode": ["tri"] * 100},
                                   "truth": {"ue_cm": est.tolist()}}]}
    assert add_road(v, "truck") is not None and v["road"]["object"] == "truck" and v["road_gt"]["source"] == "truth_smoothed"
    assert add_road({"objects": []}) is None
    from offline.presets import get_specs

    s = get_specs(["truck"], sets=["truck.road.enabled=false"])[0]
    assert s["road"]["enabled"] is False and get_specs(["drone"])[0].get("road") is None


def test_fuse_frames_fallback_and_border():
    """Слияние: основная (поза) детекция у края кадра отбрасывается; в камере без основной — запасная."""
    from offline.run import fuse_frames

    pose = [[[Detection((100, 100, 200, 150), (190, 120), 0.9, "truck", "truck_pose")],
             [Detection((0, 100, 80, 150), (5, 120), 0.9, "truck", "truck_pose")]]]
    base = [[[Detection((90, 95, 210, 160), (150, 127), 0.8, "truck", "truck")],
             [Detection((0, 95, 90, 160), (45, 127), 0.8, "truck", "truck")]]]
    fr = fuse_frames([pose, base], 1, ("truck_pose",), "fallback", 3.0)
    assert [d.source for d in fr[0][0]] == ["truck_pose"]
    assert [d.source for d in fr[0][1]] == ["truck"]


def test_secondary_offset_measurement():
    """Запасная точка со сдвигом относительно состояния: трек pivot по центрам рамок с известным сдвигом точен."""
    T = truth_traj(200)
    center = world_to_ue(T.mean(0))
    cams = [look_at_ue(loc, center, f) for loc, _, f in CAMS_UE]
    rng = np.random.default_rng(2)
    frames = []
    for k in range(200):
        v = T[min(k + 1, 199)] - T[max(k - 1, 0)]
        f = np.array([v[0], v[1], 0.0]) / np.linalg.norm(v[:2])
        Cn = T[k] - 8.0 * f  # центр на 8 м позади pivot
        row = []
        for cam in cams:
            uv = cam.project(Cn)[0] + rng.normal(0, 1.0, 2)
            s = cam.K[0, 0] * 5.0 / cam.depth(Cn)[0]
            row.append([Detection((uv[0] - s, uv[1] - s / 2, uv[0] + s, uv[1] + s / 2), tuple(uv), 0.8, "truck", "truck")])
        frames.append(row)
    cfg = TrackerConfig(fps=FPS, primary_sources=("truck_pose",), secondary_offset=(-8.0, 0.0, 0.0))
    E, _ = track_frames(cams, frames, cfg)
    err = np.linalg.norm(E - T, axis=1)
    assert np.sqrt(np.nanmean(err[60:] ** 2)) < 1.5
