"""Калибровка цепочкой по парам камер (peleng/chain.py, peleng/ext/chain.py) на синтетике."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peleng import chain as ch  # noqa: E402
from peleng.calibration import Board, BoardDetection  # noqa: E402
from peleng.geometry import CameraModel, approx_K, look_at  # noqa: E402

A3 = {"square_m": 0.07, "marker_m": 0.052}  # в комнате 6×5 м A4 издалека ненадёжна — берём A3, как советуем пользователю


def make_cam(center, target=(0, 0, 0.5), size=(1280, 720), hfov=70.0):
    R, t = look_at(np.array(center, float), np.array(target, float))
    return CameraModel(approx_K(*size, hfov), np.zeros(5), tuple(size), R, t)


def board_pose(board: Board, center, normal, roll_deg=0.0, up=(0.0, 0.0, 1.0)):
    """Поза доски (доска -> мир: X_w = R X_b + t): центр доски в center, лицевая сторона смотрит вдоль normal,
    повёрнута вокруг нормали на roll_deg."""
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    z = -n  # у доски OpenCV лицевая сторона смотрит в −z
    up = np.asarray(up, float)
    if abs(up @ n) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    y = -(up - (up @ n) * n)  # y доски — вниз по картинке
    y /= np.linalg.norm(y)
    R = np.column_stack([np.cross(y, z), y, z]) @ cv2.Rodrigues(np.array([0.0, 0.0, np.radians(roll_deg)]))[0]
    return R, np.asarray(center, float) - R @ board.center_b


def render_board_at(board: Board, cam: CameraModel, R_wb, t_wb, pps=300, margin=60, ss=3, blur=0.7, noise=3.0, rng=None):
    """Как render_board из test_core, но доска в произвольной позе (R_wb, t_wb) + лёгкое размытие и шум, как у реальной оптики."""
    if ss > 1:
        big = CameraModel(cam.K * np.array([[ss], [ss], [1]]), cam.dist, (cam.size[0] * ss, cam.size[1] * ss), cam.R, cam.t)
        img = render_board_at(board, big, R_wb, t_wb, pps, margin, ss=1, blur=0, noise=0)
        img = cv2.resize(img, cam.size, interpolation=cv2.INTER_AREA)
        if blur:
            img = cv2.GaussianBlur(img, (0, 0), blur)
        if noise:
            rng = rng or np.random.default_rng(0)
            img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
        return img
    img = board.image(pps, margin)
    s = board.square_m / pps
    S = np.array([[s, 0, -margin * s], [0, s, -margin * s], [0, 0, 1]])
    R_cb = cam.R @ R_wb
    t_cb = cam.R @ t_wb + cam.t
    H = cam.K @ np.column_stack([R_cb[:, 0], R_cb[:, 1], t_cb]) @ S
    out = cv2.warpPerspective(img, H, cam.size, borderValue=200)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def room_scene(board: Board):
    """Комната 6×5 м, 4 камеры у стен по кругу смотрят внутрь (одна — 1920×1080). Мир = доска на полу в центре."""
    cams = {}
    for i, (x, y) in enumerate([(2.9, 0.0), (0.0, 2.4), (-2.9, 0.0), (0.0, -2.4)]):
        cams[f"c{i + 1}"] = make_cam([x, y, 2.0], (0, 0, 0.5), size=(1920, 1080) if i == 1 else (1280, 720))
    cent = {c: m.center for c, m in cams.items()}
    poses = [(board.R_wb, board.t_wb)]  # 0 — опорный: доска на полу
    # дальше — доска в руках между соседними камерами, лицом «между ними», в разных местах и поворотах:
    # (a, b, доля пути от центра к середине между камерами, высота, поворот доски вокруг нормали, °)
    for a, b, r, h, roll in [("c1", "c2", 0.6, 1.0, 0), ("c2", "c3", 0.6, 1.3, 25), ("c3", "c4", 0.6, 1.0, -20),
                             ("c4", "c1", 0.6, 0.8, 10), ("c1", "c2", 0.45, 1.4, -30), ("c3", "c4", 0.45, 1.2, 15)]:
        p = (cent[a] + cent[b]) / 2 * r
        p[2] = h
        poses.append(board_pose(board, p, toward(p, cent[a], cent[b]), roll))
    return cams, poses


def toward(p, ca, cb):
    """Нормаль доски в точке p: биссектриса направлений на камеры a и b (обе видят доску под одинаковым углом)."""
    n = (ca - p) / np.linalg.norm(ca - p) + (cb - p) / np.linalg.norm(cb - p)
    return n / np.linalg.norm(n)


def facing_deg(board: Board, cam: CameraModel, R_wb, t_wb) -> float:
    """Угол между нормалью лицевой стороны доски и направлением на камеру."""
    n = -R_wb[:, 2]
    c = R_wb @ board.center_b + t_wb
    v = cam.center - c
    return float(np.degrees(np.arccos(np.clip(n @ v / np.linalg.norm(v), -1, 1))))


def project_obs(board: Board, cams: dict, poses, noise_px=0.0, max_dist=3.3, max_angle=65.0, seed=0):
    """Быстрые наблюдения без рендера: проекции углов + шум; камера «видит» доску ближе max_dist и под углом < max_angle."""
    rng = np.random.default_rng(seed)
    obs = []
    for k, (R, t) in enumerate(poses):
        X = board.all_corners @ R.T + t
        for c, m in cams.items():
            if (m.depth(X) <= 0.1).any() or np.linalg.norm(m.center - X.mean(0)) > max_dist or facing_deg(board, m, R, t) > max_angle:
                continue
            px = m.project(X)
            inside = (px[:, 0] > 5) & (px[:, 0] < m.size[0] - 5) & (px[:, 1] > 5) & (px[:, 1] < m.size[1] - 5)
            if inside.sum() < 6:
                continue
            ids = np.nonzero(inside)[0]
            obs.append(ch.Observation(k, c, px[ids] + rng.normal(0, noise_px, (len(ids), 2)), ids, m.K, m.dist, m.size))
    return obs


def center_errors(res, cams) -> dict:
    return {c: float(np.linalg.norm(x["center"] - cams[c].center)) for c, x in res.cams.items()}


# ---------------------------------------------------------------------------


def test_chain_on_rendered_room():
    """Главный сценарий: настоящая детекция ChArUco на отрендеренных кадрах, ни один снимок не виден всем камерам."""
    board = Board(A3)
    cams, poses = room_scene(board)
    rng = np.random.default_rng(1)
    obs = []
    for k, (R, t) in enumerate(poses):
        for c, m in cams.items():
            if (m.depth(board.all_corners @ R.T + t) <= 0.1).any():
                continue
            d = board.detect(render_board_at(board, m, R, t, rng=rng))
            if d is not None:
                obs.append(ch.Observation(k, c, d.corners, d.ids, m.K, m.dist, m.size))
    per_snap = {k: {o.cam for o in obs if o.snap == k} for k in range(len(poses))}
    assert 1 <= len(per_snap[0]) <= 2, per_snap  # доску на полу видят 1–2 камеры
    assert all(len(v) <= 2 for v in per_snap.values()), "ни одно положение доски не должно быть видно всем"
    assert all(len(v) == 2 for k, v in per_snap.items() if k > 0)

    res = ch.solve(board, obs)
    assert res.ok and not res.unlinked and sorted(res.cams) == sorted(cams)
    err = center_errors(res, cams)
    assert max(err.values()) < 0.03, err
    assert res.reproj_px < 1.0
    assert all(x["reproj_px"] < 1.0 for x in res.cams.values())
    assert res.reproj_px <= res.chain_reproj_px + 1e-9
    # ориентация тоже
    for c, x in res.cams.items():
        dR = x["R"] @ cams[c].R.T
        assert np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))) < 0.5
    # оценка неопределённости — того же порядка, что реальная ошибка
    for c, x in res.cams.items():
        assert 0.001 < x["center_std_norm_m"] < 0.05
    # поза доски в снимках: опорный — ровно на полу
    assert np.allclose(res.snaps[0]["corners_world"][:, 2], 0)
    s = res.summary()
    assert set(s["cams"]) == set(cams) and s["reproj_px"] < 1.0


def pair_poses(board: Board, cams: dict, per_pair: int, anchor_shift=(0.0, 0.0, 0.0), seed=42):
    """Опорный снимок (доска на полу, сдвинута на anchor_shift) + per_pair случайных поз доски между каждой парой соседних камер."""
    rng = np.random.default_rng(seed)
    cent = {c: m.center for c, m in cams.items()}
    poses = [(board.R_wb, board.t_wb + np.asarray(anchor_shift, float))]
    for _ in range(per_pair):
        for a, b in [("c1", "c2"), ("c2", "c3"), ("c3", "c4"), ("c4", "c1")]:
            p = (cent[a] + cent[b]) / 2 * rng.uniform(0.45, 0.65)
            p[2] = rng.uniform(0.8, 1.5)
            poses.append(board_pose(board, p, toward(p, cent[a], cent[b]), rng.uniform(-30, 30)))
    return poses


def shifted(cams: dict, shift) -> dict:
    """Истинные камеры в системе координат, где опорная доска сдвинута на shift (мир = опорная доска)."""
    shift = np.asarray(shift, float)
    return {c: CameraModel(m.K, m.dist, m.size, m.R, m.t + m.R @ shift) for c, m in cams.items()}


def baseline_errors(res, truth) -> list[float]:
    """Ошибки расстояний между камерами — «форма» расстановки, без общего сдвига/поворота мира."""
    cs = sorted(res.cams)
    return [abs(np.linalg.norm(res.cams[a]["center"] - res.cams[b]["center"]) - np.linalg.norm(truth[a].center - truth[b].center))
            for i, a in enumerate(cs) for b in cs[i + 1:]]


@pytest.mark.parametrize("anchor_shift,n_anchor", [((0.0, 0.0, 0.0), 2), ((0.0, 0.9, 0.0), 1)])
def test_refine_beats_pure_chain_with_noise(anchor_shift, n_anchor):
    """Шум углов 0.4 px, по 3 снимка на пару соседних камер. Совместное уточнение точнее чистой цепочки.

    Абсолютную ошибку ограничивает опорный снимок: мир задаётся одним видом доски на полу издалека (~3 м),
    и его ошибка (в основном наклон плоскости) переносится на все камеры — так же, как у обычной калибровки одним снимком.
    А взаимное положение камер (расстояния между ними) уточнение улучшает в разы."""
    board = Board(A3)
    cams, _ = room_scene(board)
    poses = pair_poses(board, cams, 3, anchor_shift)
    truth = shifted(cams, anchor_shift)
    e = {"chain": [], "ba": [], "chain_base": [], "ba_base": []}
    for seed in range(5):
        obs = project_obs(board, cams, poses, noise_px=0.4, seed=seed)
        assert len({o.cam for o in obs if o.snap == 0}) == n_anchor
        r0 = ch.solve(board, obs, refine=False)
        r1 = ch.solve(board, obs, refine=True)
        assert not r0.unlinked and not r1.unlinked
        assert r1.reproj_px < r0.reproj_px
        e["chain"] += center_errors(r0, truth).values()
        e["ba"] += center_errors(r1, truth).values()
        e["chain_base"] += baseline_errors(r0, truth)
        e["ba_base"] += baseline_errors(r1, truth)
    cm = {k: (np.mean(v) * 100, np.max(v) * 100) for k, v in e.items()}
    print("\nопорный снимок видят %d кам.: " % n_anchor + "; ".join(f"{k}: средняя {a:.2f} см, макс {b:.2f} см" for k, (a, b) in cm.items()))
    assert np.mean(e["ba"]) < np.mean(e["chain"])
    assert np.mean(e["ba_base"]) < 0.5 * np.mean(e["chain_base"])
    assert np.max(e["ba_base"]) < 0.02
    if n_anchor == 2:
        assert np.mean(e["ba"]) < 0.025 and np.max(e["ba"]) < 0.05


def test_extra_floor_snapshots_fix_world_tilt():
    """Опорную доску видит одна камера издалека → мир чуть «наклонён». Снимки доски на полу перед каждой камерой
    (их видит хотя бы одна камера) фиксируют плоскость пола — абсолютная ошибка падает в разы."""
    board = Board(A3)
    cams, _ = room_scene(board)
    shift = (0.0, 0.9, 0.0)
    truth = shifted(cams, shift)
    poses = pair_poses(board, cams, 3, shift)
    floor = set()
    for m in cams.values():
        C = m.center
        p = np.array([*(C[:2] - 1.8 * C[:2] / np.linalg.norm(C[:2])), 0.0])  # 1.8 м от камеры по полу к центру
        R = board.R_wb @ cv2.Rodrigues(np.array([0.0, 0.0, 0.3]))[0]  # повёрнута вокруг вертикали
        floor.add(len(poses))
        poses.append((R, p - R @ board.center_b + np.array([0.0, 0.0, board.t_wb[2]])))
    e_no, e_floor = [], []
    for seed in range(4):
        obs = project_obs(board, cams, poses, noise_px=0.4, seed=seed)
        assert {o.snap for o in obs} >= floor
        e_no += center_errors(ch.solve(board, obs), truth).values()
        res = ch.solve(board, obs, floor_snaps=floor)
        e_floor += center_errors(res, truth).values()
        for k in floor:  # доска в этих снимках действительно легла на пол
            assert np.abs(res.snaps[k]["corners_world"][:, 2]).max() < 0.01
    print(f"\nбез снимков на полу: средняя {np.mean(e_no) * 100:.2f} см, макс {np.max(e_no) * 100:.2f} см; "
          f"с ними: средняя {np.mean(e_floor) * 100:.2f} см, макс {np.max(e_floor) * 100:.2f} см")
    assert np.mean(e_floor) < 0.6 * np.mean(e_no)
    assert np.max(e_floor) < 0.05


def test_unlinked_camera_untouched_and_hints():
    board = Board(A3)
    cams, poses = room_scene(board)
    obs = project_obs(board, cams, poses)
    # пятая камера видела доску только вместе с другой несвязанной (снимок 99 без связи с миром)
    c5 = make_cam([8.0, 1.0, 1.8], (10.0, 0.0, 1.0))  # в соседней комнате
    c6 = make_cam([8.0, -1.0, 1.8], (10.0, 0.0, 1.0))
    far = board_pose(board, [10.0, 0.0, 1.0], [-1.0, 0.0, 0.0])
    extra = project_obs(board, {"c5": c5, "c6": c6}, [far])
    assert extra
    for o in extra:
        o.snap = 99
    res = ch.solve(board, obs + extra)
    assert res.ok and set(res.unlinked) == {"c5", "c6"}
    assert "c5" not in res.cams and any("c5" in w for w in res.warnings)
    g = ch.graph_summary(obs + extra, list(cams) + ["c5", "c6", "c7"])
    assert g["cams"]["c5"]["linked"] is False and g["cams"]["c7"]["snapshots"] == 0
    assert {"a": "c5", "b": "c6", "n": 1} in g["pairs"]
    assert any("c7" in h and "Переместите доску" in h for h in g["hints"])
    # подсказка с учётом прежних положений — ближайшая привязанная камера
    g = ch.graph_summary(obs, list(cams) + ["c5"], prior_centers={**{c: m.center for c, m in cams.items()}, "c5": c5.center})
    assert any("между c5 и" in h and ("c1" in h or "c2" in h) for h in g["hints"])
    # без опорного снимка — подсказка положить доску на пол
    g = ch.graph_summary([o for o in obs if o.snap != 0], list(cams))
    assert not g["linked"] and "на пол" in g["hints"][0]
    assert ch.solve(board, [o for o in obs if o.snap != 0]).ok is False


def test_outlier_observation_dropped():
    """Доска сдвинулась между кадрами одной камеры: это наблюдение отбрасывается, остальные камеры точны.
    (Заметить такое можно, только если есть избыточность — несколько снимков на пару.)"""
    board = Board(A3)
    cams, _ = room_scene(board)
    poses = pair_poses(board, cams, 2)
    obs = project_obs(board, cams, poses, noise_px=0.3, seed=3)
    good = ch.solve(board, obs)
    assert not good.dropped
    k_bad = next(o.snap for o in obs if o.snap > 0 and o.cam == "c3")
    R, t = poses[k_bad]
    Rm = cv2.Rodrigues(np.array([0.0, 0.0, 0.15]))[0]  # c3 «увидела» доску сдвинутой на 15 см и повёрнутой на 9°
    bad = project_obs(board, {"c3": cams["c3"]}, [(Rm @ R, t + np.array([0.15, 0.0, 0.05]))], max_dist=99, max_angle=89)[0]
    bad.snap = k_bad
    obs = [o for o in obs if not (o.snap == k_bad and o.cam == "c3")] + [bad]
    res = ch.solve(board, obs)
    assert res.ok and any(d["snap"] == k_bad for d in res.dropped), res.dropped
    err = center_errors(res, cams)
    assert max(err.values()) < 0.03, err
    assert res.reproj_px < 1.0
    assert any("отброшено" in w for w in res.warnings)


def test_observation_filters():
    board = Board(A3)
    cam = make_cam([0.0, -2.0, 1.5])
    X = board.all_corners @ board.R_wb.T + board.t_wb
    px = cam.project(X)
    ids = np.arange(len(X))
    assert ch.observation_ok(ch.Observation(0, "a", px, ids, cam.K, cam.dist, cam.size), board)
    row = ids[:4]  # 4 угла в одну строку — мало и на одной линии
    assert not ch.observation_ok(ch.Observation(0, "a", px[row], row, cam.K, cam.dist, cam.size), board)
    p = ch.pnp(board, ch.Observation(0, "a", px, ids, cam.K, cam.dist, cam.size))
    R_cb, t_cb, err, amb = p
    assert err < 1e-3 and amb < 0.5 and np.allclose(R_cb, cam.R @ board.R_wb, atol=1e-6)


# ---------------------------------------------------------------------------
# REST


def _app(tmp_path, n_cams=3):
    from peleng.config import load_config
    from peleng.server import create_app

    cfg = load_config()
    cfg["demo"] = {"cameras": n_cams, "people": 1}
    cfg["chain"] = {"collect_s": 0.2}
    return create_app(cfg, demo=True, data_dir=tmp_path)


def test_rest_demo_no_board(tmp_path):
    """В демо доски в кадрах нет: снимок честно отвечает «доска не видна», состояние сбрасывается."""
    from fastapi.testclient import TestClient

    app = _app(tmp_path)
    with TestClient(app) as client:
        s = client.get("/api/chain").json()
        assert s["type"] == "chain" and s["active"] is False and s["snaps"] == []
        assert client.post("/api/chain/snapshot").json()["ok"] is False  # не начато
        assert client.post("/api/chain/auto", json={"on": True}).status_code == 409
        s = client.post("/api/chain/start").json()
        assert s["active"] and "на пол" in s["graph"]["hints"][0]
        time.sleep(0.5)  # демо-камеры выходят в онлайн
        r = client.post("/api/chain/snapshot").json()
        assert r["ok"] is False and "не видна" in r["message"]
        assert client.post("/api/chain/solve").json()["ok"] is False
        assert client.post("/api/chain/apply").json()["ok"] is False
        assert client.post("/api/chain/snapshot/0/delete").status_code == 400
        assert client.post("/api/chain/auto", json={"on": True}).json()["auto"] is True
        s = client.post("/api/chain/cancel").json()
        assert s["active"] is False and s["auto"] is False and s["snaps"] == []
        assert "chain.js" in client.get("/api/ext").json()
        app.state.pipeline.stop()
        time.sleep(0.3)
    assert not list((tmp_path / "calib").glob("*.json"))


class _FakeFrame:
    def __init__(self, size):
        self.image = np.zeros((size[1], size[0], 3), np.uint8)


def test_rest_full_flow_with_fake_board(tmp_path):
    """Весь процесс через REST: подменяем сбор кадров доски проекциями (демо-камеры известны) → Применить."""
    from fastapi.testclient import TestClient

    app = _app(tmp_path, n_cams=3)
    reg = app.state.registry
    board = app.state.board
    truth = {c: reg.model(c) for c in reg.sources}
    cur = {"pose": None}

    def fake_collect(ids, duration=1.0):
        R, t = cur["pose"]
        out = {}
        for o in project_obs(board, {c: truth[c] for c in ids}, [(R, t)], noise_px=0.05, max_dist=99, max_angle=75, seed=len(ids)):
            d = BoardDetection(o.corners, o.ids, board.all_corners[o.ids])
            out[o.cam] = [(_FakeFrame(truth[o.cam].size), d)] * 3
        return out

    centers = {c: m.center for c, m in truth.items()}
    with TestClient(app) as client:
        app.state.collect_board = fake_collect
        # испортим текущие положения камер — «Применить» должно вернуть истинные
        for c, m in truth.items():
            reg.set_model(c, CameraModel(m.K, m.dist, m.size, m.R, m.t + 0.3, dict(m.meta)), save=False)
        client.post("/api/chain/start")
        time.sleep(0.5)
        cur["pose"] = (board.R_wb, board.t_wb)
        r = client.post("/api/chain/snapshot").json()
        assert r["ok"] and r["snapshot"]["anchor"], r
        c1, c2, c3 = sorted(truth)
        for a, b, h in [(c1, c2, 1.0), (c2, c3, 1.2), (c1, c2, 1.4), (c2, c3, 0.9)]:
            mid = (centers[a] + centers[b]) / 2
            p = mid * 0.5
            p[2] = h
            n = mid - p
            n[2] = 0.3
            cur["pose"] = board_pose(board, p, n)
            r = client.post("/api/chain/snapshot").json()
            assert r["ok"], r
        # доска на полу в другом месте, видна хотя бы одной камере — флаг floor
        cur["pose"] = (board.R_wb, board.t_wb + np.array([0.6, -0.4, 0.0]))
        r = client.post("/api/chain/snapshot", json={"floor": True}).json()
        assert r["ok"] and r["snapshot"]["floor"], r
        s = client.get("/api/chain").json()
        assert len(s["snaps"]) == 6 and not s["graph"]["unlinked"]
        assert [x["k"] for x in client.post("/api/chain/snapshot/4/delete").json()["snaps"]] == [0, 1, 2, 3, 5]
        pre = client.post("/api/chain/solve").json()
        assert pre["ok"] and pre["reproj_px"] < 1.0 and len(pre["snaps"]) == 5
        assert max(abs(p[2]) for p in pre["snaps"]["5"]["corners_world"]) < 0.01
        # оценка — ещё не применена: модели прежние (испорченные)
        assert np.linalg.norm(reg.model(c1).center - centers[c1]) > 0.1
        r = client.post("/api/chain/apply").json()
        assert r["ok"] and r["saved"] is False and all(x["ok"] for x in r["cameras"].values())
        for c in truth:
            m = reg.model(c)
            assert np.linalg.norm(m.center - centers[c]) < 0.02
            assert m.meta["extrinsics_method"] == "chain" and m.meta["extrinsics_chain"]["hops"] >= 1
        s = client.get("/api/chain").json()
        assert s["active"] is False and s["applied"]["cameras"] == sorted(truth)
        app.state.pipeline.stop()
        time.sleep(0.3)
    assert not list((tmp_path / "calib").glob("*.json")), "демо не должно писать калибровку на диск"
