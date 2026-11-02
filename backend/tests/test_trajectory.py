"""Восстановление траекторий после записи (peleng/trajectory.py) — на синтетике, без камер."""
import csv
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peleng import trajectory as tj  # noqa: E402
from peleng.demo import DemoWorld  # noqa: E402

FPS = 15.0


def truth_fn(world: DemoWorld, k: int = 0):
    """Истинный центр таза человека k из DemoWorld (центр на полу + 0.95 м)."""
    return lambda t: np.array([world.people(x)[k][0] + [0, 0, 0.95] for x in np.atleast_1d(t)])


def person_item(pid, pos, mode="tri", predicted=False, vel=(0, 0, 0)):
    return {"id": pid, "pos": [round(float(v), 4) for v in pos], "floor": [round(float(pos[0]), 4), round(float(pos[1]), 4), 0.0],
            "vel": list(vel), "speed": 0.0, "mode": mode, "cams": ["demo1", "demo2"],
            "reproj_err_px": 1.5 if mode == "tri" else None, "predicted": predicted, "kp3d": None}


def write_session(path: Path, frames: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "hello", "mode": "demo", "server_time": frames[0]["t"] if frames else 0, "cameras": []}) + "\n")
        for i, fr in enumerate(frames):
            f.write(json.dumps({"type": "frame", "seq": i + 1, "detections": {}, "stats": {}, **fr}) + "\n")
    return path


def noisy_track(truth, t, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return truth(t) + rng.normal(0, noise, (len(t), 3))


def raw_track(pid, t, pos, kind="person", mode="tri"):
    n = len(t)
    return tj.RawTrack(kind, pid, np.asarray(t, float), np.asarray(pos, float), [mode] * n, np.zeros(n, bool),
                       np.full(n, np.nan), np.full((n, 3), np.nan), np.full((n, 3), np.nan))


def test_restore_reduces_error_and_fills_gap():
    world = DemoWorld(n_cams=1, n_people=1)
    truth = truth_fn(world)
    t0 = world.t0
    t = t0 + np.arange(0, 12, 1 / FPS)
    gap = (t > t0 + 5.0) & (t < t0 + 5.5)  # пропуск 0.5 с
    gap2 = (t > t0 + 8.0) & (t < t0 + 8.3)
    t_meas = t[~gap & ~gap2]
    z = noisy_track(truth, t_meas, noise=0.05, seed=1)
    # пара «прогнозных» точек в пропуске — они не должны считаться измерениями
    t_pred = t[gap][:3]
    frames = [{"t": float(tt), "persons": [person_item(1, p)]} for tt, p in zip(t_meas, z)]
    frames += [{"t": float(tt), "persons": [person_item(1, truth(tt)[0] + [0.5, 0.5, 0], predicted=True)]} for tt in t_pred]
    frames.sort(key=lambda f: f["t"])
    res = tj.restore_tracks(_load(frames), tj.settings())
    assert len(res) == 1
    r = res[0]
    err = np.linalg.norm(r.pos - truth(r.t), axis=1)
    raw_err = np.linalg.norm(z - truth(t_meas), axis=1)
    interp = np.array([s == "interpolated" for s in r.source])
    gap_err = err[interp & (r.t > t0 + 5.0) & (r.t < t0 + 5.5)]
    print(f"\nсырые: ср. ошибка {raw_err.mean() * 100:.1f} см (RMS {np.sqrt((raw_err**2).mean()) * 100:.1f}); "
          f"восстановленные: {err.mean() * 100:.1f} см (RMS {np.sqrt((err**2).mean()) * 100:.1f}); "
          f"в пропуске 0.5 с: макс {gap_err.max() * 100:.1f} см; СКО ср. {r.std.mean() * 100:.1f} см")
    assert err.mean() < 0.6 * raw_err.mean()
    assert len(gap_err) >= 12 and gap_err.max() < 0.10
    assert r.metrics["gaps_filled"] == 2 and 0.75 < r.metrics["gaps_filled_s"] < 1.0
    assert r.metrics["points_predicted"] == 3
    # равномерная сетка 30 Гц
    assert np.allclose(np.diff(r.t), 1 / 30, atol=1e-6)
    # путь по сглаженной траектории близок к истинному, по сырым — сильно завышен шумом
    tt = np.arange(t_meas[0], t_meas[-1], 0.01)
    true_path = np.linalg.norm(np.diff(truth(tt)[:, :2], axis=0), axis=1).sum()
    assert abs(r.metrics["path_m"] - true_path) < 0.05 * true_path
    assert r.metrics["path_raw_m"] > 1.3 * true_path
    # σ в пропуске больше, чем на измерениях
    assert r.std[interp].mean() > r.std[~interp].mean()


def _load(frames):
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        _, raw = tj.load_session(write_session(Path(d) / "s.jsonl", frames))
    return raw


def test_mono_weighted_less_than_tri():
    world = DemoWorld(n_cams=1, n_people=1)
    truth = truth_fn(world)
    t = world.t0 + np.arange(0, 6, 1 / FPS)
    rng = np.random.default_rng(3)
    z = truth(t) + rng.normal(0, 0.03, (len(t), 3))
    mono = rng.random(len(t)) < 0.3
    z[mono] += rng.normal(0, 0.4, (mono.sum(), 3))  # грубые mono-точки
    tr = raw_track(1, t, z)
    tr.mode = ["mono" if m else "tri" for m in mono]
    r = tj.restore_track(tr, tj.settings())
    tr_eq = raw_track(1, t, z)  # те же точки, но все считаются tri
    r_eq = tj.restore_track(tr_eq, tj.settings())
    e = np.linalg.norm(r.pos - truth(r.t), axis=1).mean()
    e_eq = np.linalg.norm(r_eq.pos - truth(r_eq.t), axis=1).mean()
    assert e < 0.7 * e_eq


def test_stitch_same_person():
    world = DemoWorld(n_cams=1, n_people=1)
    truth = truth_fn(world)
    t0 = world.t0
    ta = t0 + np.arange(0, 4, 1 / FPS)
    tb = t0 + np.arange(4.6, 9, 1 / FPS)  # человек пропал на 0.6 с, трекер дал новый ID
    frames = [{"t": float(t), "persons": [person_item(1, p)]} for t, p in zip(ta, noisy_track(truth, ta, 0.03, 1))]
    # живой трекер ещё ~0.5 с выдаёт прогноз старого трека
    frames += [{"t": float(t), "persons": [person_item(1, truth(t)[0] + [0.2, 0, 0], predicted=True)]} for t in t0 + np.arange(4.0, 4.5, 1 / FPS)]
    frames += [{"t": float(t), "persons": [person_item(7, p)]} for t, p in zip(tb, noisy_track(truth, tb, 0.03, 2))]
    frames.sort(key=lambda f: f["t"])
    raw = _load(frames)
    assert sorted(tr.id for tr in raw) == [1, 7]
    res = tj.restore_tracks(raw, tj.settings())
    assert len(res) == 1 and res[0].merged_ids == [1, 7]
    r = res[0]
    err = np.linalg.norm(r.pos - truth(r.t), axis=1)
    assert err.max() < 0.12 and r.metrics["gaps_filled"] >= 1
    # с max_gap_s меньше пропуска — не сшиваются
    assert len(tj.restore_tracks(raw, tj.settings(max_gap_s=0.3))) == 2


def test_no_stitch_different_people():
    t0 = 1000.0
    # A идёт по +x и пропадает; через 0.5 с в 2 м сбоку появляется B (другой человек)
    ta = t0 + np.arange(0, 3, 1 / FPS)
    pa = np.column_stack([(ta - t0) * 1.0, np.zeros_like(ta), np.full_like(ta, 0.95)])
    tb = t0 + np.arange(3.5, 6, 1 / FPS)
    pb = np.column_stack([np.full_like(tb, 3.5), 2.0 + (tb - tb[0]) * 0.8, np.full_like(tb, 0.95)])
    # C и D видны одновременно — точно разные люди, даже рядом
    tc = t0 + np.arange(0, 3, 1 / FPS)
    pc = np.column_stack([np.full_like(tc, -2.0), np.zeros_like(tc), np.full_like(tc, 0.95)])
    td = t0 + np.arange(1, 4, 1 / FPS)
    pd = pc[: len(td)] + [0.3, 0, 0]
    raw = [raw_track(1, ta, pa), raw_track(2, tb, pb), raw_track(3, tc, pc), raw_track(4, td, pd)]
    res = tj.restore_tracks(raw, tj.settings())
    assert sorted(r.merged_ids for r in res) == [[1], [2], [3], [4]]
    # а объект того же места, но другого kind с человеком не сшивается
    obj = raw_track(5, tb, pa[-1] + (tb - ta[-1])[:, None] * [1.0, 0, 0], kind="object")
    res = tj.restore_tracks([raw_track(1, ta, pa), obj], tj.settings())
    assert sorted((r.kind, r.merged_ids[0], len(r.merged_ids)) for r in res) == [("object", 5, 1), ("person", 1, 1)]


def test_long_gap_not_filled():
    t = np.concatenate([np.arange(0, 2, 1 / FPS), np.arange(5, 7, 1 / FPS)]) + 50.0
    pos = np.column_stack([t * 0.1, np.zeros_like(t), np.full_like(t, 0.95)])
    r = tj.restore_track(raw_track(1, t, pos), tj.settings(max_gap_s=1.0))
    assert not ((r.t > 52.1) & (r.t < 54.9)).any()
    assert r.metrics["breaks"] == 1 and r.metrics["gaps_filled"] == 0
    assert abs(r.metrics["duration_s"] - 2 * (2 - 1 / FPS)) < 0.1


def test_session_with_objects_and_csv_roundtrip(tmp_path):
    t0 = time.time()
    frames = []
    for i in range(60):
        t = t0 + i / FPS
        persons = [person_item(3, [0.5 + 0.02 * i, 0.0, 0.95], mode="tri" if i % 5 else "mono")]
        objects = [{"id": 101, "kind": "object", "cls": "sports ball", "pos": [1.0, 0.1 * i / FPS * 15, 0.3], "vel": [0, 1.5, 0],
                    "speed": 1.5, "mode": "tri", "cams": ["a", "b"], "reproj_err_px": 1.2, "predicted": False, "pos_std_m": [0.01, 0.01, 0.02]}]
        frames.append({"t": round(t, 4), "persons": persons, "objects": objects})
    frames.append({"t": t0 + 4.1, "persons": []})
    p = write_session(tmp_path / "session_test.jsonl", frames)
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"type": "frame", "t": 12')  # оборванная последняя строка (запись не закрыта)
    res = tj.restore_session(p)
    assert res.hello and res.hello["type"] == "hello"
    assert {(tr.kind, tr.id) for tr in res.raw} == {("person", 3), ("object", 101)}
    obj = next(tr for tr in res.raw if tr.kind == "object")
    assert obj.cls == "sports ball" and np.allclose(obj.pos_std[0], [0.01, 0.01, 0.02])
    assert {(r.kind, r.id) for r in res.restored} == {("person", 3), ("object", 101)}
    # скорость объекта — 3D, ~1.5 м/с
    ro = next(r for r in res.restored if r.kind == "object")
    assert abs(ro.metrics["speed_mean_mps"] - 1.5) < 0.05

    text = tj.to_csv(res)
    rows = list(csv.DictReader(io.StringIO(text)))
    assert list(rows[0].keys()) == tj.CSV_COLUMNS
    n = sum(len(r.t) for r in res.restored)
    assert len(rows) == n
    person_rows = [r for r in rows if r["kind"] == "person"]
    rp = next(r for r in res.restored if r.kind == "person")
    xyz = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in person_rows])
    assert np.allclose(xyz, rp.pos, atol=1e-4)
    assert all(r["source"] in ("measured", "interpolated") and float(r["std"]) > 0 for r in rows)
    assert np.allclose([float(r["t"]) for r in person_rows], rp.t, atol=1e-4)

    raw_rows = list(csv.DictReader(io.StringIO(tj.to_csv(res, raw=True))))
    assert len(raw_rows) == 120 and {r["source"] for r in raw_rows} == {"tri", "mono"}
    assert {r["std"] for r in raw_rows if r["kind"] == "person"} == {""}
    assert all(float(r["std"]) > 0 for r in raw_rows if r["kind"] == "object")

    j = json.loads(json.dumps(tj.to_json(res)))
    assert j["data"] == "restored" and len(j["tracks"]) == 2 and "metrics" in j["tracks"][0]
    jr = tj.to_json(res, raw=True)
    assert sum(len(tr["t"]) for tr in jr["tracks"]) == 120


def test_api_and_end_to_end_demo(tmp_path):
    """Сквозной сценарий: демо-сервер → запись сессии → восстановление через API → CSV."""
    from fastapi.testclient import TestClient

    from peleng.config import load_config
    from peleng.server import create_app

    app = create_app(load_config(), demo=True, data_dir=tmp_path)
    with TestClient(app) as c:
        c.post("/api/record/start")
        time.sleep(3.0)
        name = Path(c.post("/api/record/stop").json()["file"]).name
        lst = c.get("/api/trajectory/sessions").json()
        assert [s["name"] for s in lst] == [name] and lst[0]["size"] > 0
        r = c.get("/api/trajectory/restore", params={"session": name, "max_gap_s": 1.5, "rate_hz": 30})
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["tracks"] >= 1 and j["restored"] and j["raw"]
        assert all(it["duration_s"] > 0 for it in j["items"])
        csv_r = c.get("/api/trajectory/export", params={"session": name, "fmt": "csv", "data": "restored"})
        assert csv_r.status_code == 200 and "attachment" in csv_r.headers["content-disposition"]
        rows = list(csv.DictReader(io.StringIO(csv_r.text)))
        assert rows and list(rows[0].keys()) == tj.CSV_COLUMNS
        assert c.get("/api/trajectory/export", params={"session": name, "fmt": "json", "data": "raw"}).json()["data"] == "raw"
        # восстановленные линии гладкие: без скачков между соседними точками сетки
        for tr in j["restored"]:
            p = np.array(tr["pos"])
            if len(p) > 3:
                step = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
                assert step.max() < 0.2  # 30 Гц, люди ходят < 1 м/с
        # имена вне папки sessions не принимаются
        assert c.get("/api/trajectory/restore", params={"session": "../config.json"}).status_code == 400
        assert c.get("/api/trajectory/restore", params={"session": "..\\x.jsonl"}).status_code == 400
        assert c.get("/api/trajectory/restore", params={"session": "nope.jsonl"}).status_code == 404
        assert c.get("/api/trajectory/export", params={"session": name, "fmt": "xml"}).status_code == 400
        assert c.get("/api/trajectory/restore", params={"session": name, "rate_hz": 0}).status_code == 400
