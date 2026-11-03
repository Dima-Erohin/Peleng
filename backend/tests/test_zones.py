"""Зоны, линии подсчёта, тепловая карта (peleng/zones.py) — на синтетике, без камер."""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peleng import zones as zn  # noqa: E402

DT = 1 / 15
ZONE = {"id": "z1", "name": "Дверь", "color": "#58a6ff", "type": "rect", "points": [[1, -1], [2, -1], [2, 1], [1, 1]]}
LINE = {"id": "l1", "name": "Проход", "color": "#f0883e", "a": [0.0, -1.0], "b": [0.0, 1.0]}


def frame(t, items, kind="persons"):
    return {"type": "frame", "t": round(t, 4), kind: [
        {"id": i, "pos": [x, y, 0.95], "floor": [x, y, 0.0], "predicted": pred, "mode": "tri"} for i, x, y, pred in items]}


def walk(eng, path, t0=0.0, tid=1, noise=0.0, seed=0):
    """path(t) -> (x, y); возвращает все события."""
    rng = np.random.default_rng(seed)
    ev = []
    for k in range(int(path.T / DT) + 1):
        t = t0 + k * DT
        x, y = path(k * DT)
        ev += eng.process(frame(t, [(tid, x + rng.normal(0, noise), y + rng.normal(0, noise), False)]))
    return ev


def line_path(x0, y0, x1, y1, T):
    f = lambda s: (x0 + (x1 - x0) * s / T, y0 + (y1 - y0) * s / T)  # noqa: E731
    f.T = T
    return f


def test_enter_exit_once_and_dwell():
    eng = zn.Engine([ZONE], [])
    # идёт вдоль x от 0 до 3 м со скоростью 1 м/с: в зоне (x от 1 до 2) ровно 1 с
    ev = walk(eng, line_path(0, 0, 3, 0, 3.0))
    ev += eng.close_all("end")
    assert [e["type"] for e in ev] == ["enter", "exit"]
    assert ev[1]["reason"] is None
    # границу зоны пересекаем на 1.0 и 2.0, но с гистерезисом вход/выход на ±7.5 см: вход на 1.075, выход на 2.075 — dwell ≈ истинному 1 с
    assert abs(ev[1]["dwell_s"] - 1.0) < 0.1
    assert abs(ev[0]["t"] - 1.075) < 0.1
    st = eng.stats()["zones"][0]
    assert st["visits"] == 1 and st["exits"] == 1 and abs(st["mean_dwell_s"] - ev[1]["dwell_s"]) < 1e-9


def test_boundary_noise_no_chatter():
    eng = zn.Engine([ZONE], [])
    # стоит у границы зоны (x=0.97, снаружи) с шумом ±5 см; затем у выхода внутри (x=1.03)
    ev = walk(eng, line_path(0.97, 0, 0.97, 0, 3.0), noise=0.02, seed=1)
    assert ev == []
    # зашёл внутрь, стоит у границы с шумом σ=2.5 см (почти ±5 см): один enter и ни одного exit
    ev = walk(eng, line_path(1.2, 0, 1.2, 0, 1.0), t0=3.0)
    ev += walk(eng, line_path(1.02, 0, 1.02, 0, 4.0), t0=4.0, noise=0.025, seed=2)
    assert [e["type"] for e in ev] == ["enter"]
    # равномерный шум ±5 см вокруг самой границы: состояние не дребезжит
    rng = np.random.default_rng(3)
    ev2 = []
    for k in range(120):
        ev2 += eng.process(frame(10 + k * DT, [(1, 1.0 + rng.uniform(-0.05, 0.05), 0, False)]))
    assert ev2 == []


def test_min_time_filters_short_blip():
    eng = zn.Engine([ZONE], [])
    ev = []
    for k in range(60):
        x = 1.5 if 10 <= k < 13 else 0.0  # 3 кадра (0.2 с) внутри < min_time 0.3 с
        ev += eng.process(frame(k * DT, [(1, x, 0, False)]))
    assert ev == []


def test_predicted_never_starts_enter():
    eng = zn.Engine([ZONE], [])
    ev = []
    for k in range(60):  # трек «виден» только прогнозом, внутри зоны
        ev += eng.process(frame(k * DT, [(1, 1.5, 0, True)]))
    assert ev == []
    for k in range(60, 80):  # теперь измерения
        ev += eng.process(frame(k * DT, [(1, 1.5, 0, False)]))
    assert [e["type"] for e in ev] == ["enter"] and ev[0]["t"] >= 60 * DT - 1e-6
    # прогноз уносит трек из зоны, но измерения не подтверждают → выхода нет
    ev = []
    for k in range(80, 140):
        ev += eng.process(frame(k * DT, [(1, 3.0, 0, True)]))
    assert ev == []


def test_line_crossing_both_directions():
    eng = zn.Engine([], [LINE])
    ev = walk(eng, line_path(-1, 0, 1, 0, 2.0))
    assert [(e["type"], e["direction"]) for e in ev] == [("line_cross", "A→B")]  # по +x: с левой стороны вектора A→B (вверх) на правую
    ev = walk(eng, line_path(1, 0, -1, 0, 2.0), t0=3.0)
    assert [e["direction"] for e in ev] == ["B→A"]
    assert abs(ev[0]["t"] - 4.0) < 0.1  # пересекли x=0 на середине пути
    cnt = eng.stats()["lines"][0]
    assert cnt["a_to_b"] == 1 and cnt["b_to_a"] == 1 and cnt["total"] == 2


def test_line_passes_beside_ends_not_counted():
    eng = zn.Engine([], [LINE])
    ev = walk(eng, line_path(-1, 1.5, 1, 1.5, 2.0))  # мимо верхнего конца
    ev += walk(eng, line_path(-1, -1.5, 1, -1.5, 2.0), t0=3.0, tid=3)  # мимо нижнего
    ev += walk(eng, line_path(-1, 0.9, 1, 0.9, 2.0), t0=6.0, tid=2)  # а здесь внутри отрезка — считается
    assert [e["type"] for e in ev] == ["line_cross"] and ev[0]["id"] == 2


def test_line_noise_not_double_counted():
    eng = zn.Engine([], [LINE])
    rng = np.random.default_rng(5)
    ev = []
    for k in range(150):  # стоит у линии, шум ±5 см
        ev += eng.process(frame(k * DT, [(1, rng.uniform(-0.05, 0.05), 0, False)]))
    assert ev == []


def test_lost_track_closes_stay():
    eng = zn.Engine([ZONE], [])
    ev = []
    for k in range(30):
        ev += eng.process(frame(k * DT, [(1, 1.5, 0, False)]))
    assert [e["type"] for e in ev] == ["enter"]
    t_last = 29 * DT
    ev = []
    for k in range(30, 50):  # трека больше нет в сообщениях (прошло > lost_after_s)
        ev += eng.process(frame(k * DT, []))
    assert [(e["type"], e["reason"]) for e in ev] == [("exit", "lost")]
    assert abs(ev[0]["t"] - t_last) < 1e-3 and ev[0]["dwell_s"] > 1.5
    assert eng.snapshot(5.0)["inside"]["z1"] == []


def test_objects_are_tracked_separately_by_kind():
    eng = zn.Engine([ZONE], [])
    ev = []
    for k in range(30):
        m = frame(k * DT, [(5, 1.5, 0, False)])
        m["objects"] = [{"id": 5, "kind": "object", "pos": [1.5, 0, 0.1], "floor": [1.5, 0, 0], "predicted": False}]
        ev += eng.process(m)
    assert sorted((e["kind"], e["id"]) for e in ev) == [("object", 5), ("person", 5)]
    assert {w["kind"] for w in eng.snapshot(1.0)["inside"]["z1"]} == {"person", "object"}


def test_polygon_zone():
    tri = {**ZONE, "type": "poly", "points": [[0, 0], [2, 0], [0, 2]]}
    assert zn.poly_signed_dist(tri["points"], 0.5, 0.5) > 0.3
    assert zn.poly_signed_dist(tri["points"], 1.5, 1.5) < -0.3
    assert abs(zn.poly_signed_dist(tri["points"], 1.0, 0.0)) < 1e-9


def test_heatmap_sums_time_in_cells():
    room = {"size_m": [5, 4, 2.7]}
    eng = zn.Engine([], [], room=room, cell_m=0.2)
    for k in range(61):  # 4 с в одной точке (кадр 0 — dt=0)
        eng.process(frame(k * DT, [(1, 1.01, 0.51, False)]))
    for k in range(61, 91):  # 2 с в другой, прогноз не учитывается
        eng.process(frame(k * DT, [(1, -1.0, -1.0, k % 2 == 0)]))
    g = eng.heat.grid()
    assert g["bounds_from"] == "room" and g["origin"] == [-2.5, -2.0] and g["nx"] == 25 and g["ny"] == 20
    v = np.array(g["values"]).reshape(g["ny"], g["nx"])
    ix, iy = int((1.01 + 2.5) // 0.2), int((0.51 + 2.0) // 0.2)
    assert abs(v[iy, ix] - 4.0) < 0.2 and v[iy, ix] == v.max()
    assert 0.5 < v[4:7, 6:9].sum() < 1.2  # (-1,-1): только половина кадров измерения
    assert abs(v.sum() - g["total_s"]) < 0.1
    eng.heat.reset()
    assert eng.heat.grid()["total_s"] == 0.0
    # без комнаты — динамическая сетка, пустая и после точки
    dyn = zn.HeatMap(0.5)
    assert dyn.grid()["nx"] == 0
    dyn.add(0.2, 0.2, 1.0)
    dyn.add(1.7, 0.2, 2.0)
    g = dyn.grid()
    assert (g["nx"], g["ny"], g["values"]) == (4, 1, [1.0, 0.0, 0.0, 2.0])
    dyn.add(100.0, 0, 1.0)  # за пределами лимита
    assert dyn.grid()["total_s"] == 3.0


def test_store_survives_recreation(tmp_path):
    p = tmp_path / "zones.json"
    s = zn.ZoneStore(p)
    z = s.add_zone({"rect": [2, 1, 1, -1], "name": "Дверь"})
    l = s.add_line({"a": [0, 0], "b": [0, 2]})
    assert z["id"] == "z1" and z["points"] == [[1, -1], [2, -1], [2, 1], [1, 1]] and l["id"] == "l1"
    s.update("zone", "z1", {"name": "Вход", "color": "#112233"})
    s2 = zn.ZoneStore(p)
    assert s2.zones[0]["name"] == "Вход" and s2.zones[0]["color"] == "#112233" and s2.lines[0]["b"] == [0.0, 2.0]
    assert s2.add_zone({"points": [[0, 0], [1, 0], [0, 1]]})["id"] == "z2"
    assert s2.delete("zone", "z1") and not s2.delete("zone", "z1")
    assert [z["id"] for z in zn.ZoneStore(p).zones] == ["z2"]
    for bad in ({"rect": [0, 0, 0.01, 0.01]}, {"rect": [0, 0, 1]}, {}, {"points": [[0, 0], [1, 1]]}):
        try:
            s2.add_zone(bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_offline_analysis_matches_live(tmp_path):
    zones = [ZONE]
    lines = [LINE]
    live = zn.Engine(zones, lines, room={"size_m": [5, 4, 2.7]})
    msgs = []
    for k in range(int(8.0 / DT)):  # проходит туда-обратно через линию и зону
        t = k * DT
        x = 3.0 * (1 - abs(1 - t / 4.0)) - 0.5  # -0.5 → 2.5 → -0.5
        msgs.append(frame(t, [(1, x, 0.0, False), (2, -1.0, 0.5, False)]))
    ev_live = []
    for m in msgs:
        ev_live += live.process(m)
    ev_live += live.close_all("end")
    path = tmp_path / "s.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "hello", "room": {"size_m": [5, 4, 2.7]}}) + "\n")
        for m in msgs:
            f.write(json.dumps(m) + "\n")
        f.write("{оборванная строка")
    res = zn.analyze_session(path, zones, lines)
    assert res["events"] == ev_live and len(ev_live) >= 4
    types = [e["type"] for e in res["events"]]
    assert types.count("enter") == 2 and types.count("exit") == 2 and types.count("line_cross") == 2
    assert res["summary"]["zones"][0]["visits"] == 2 and res["summary"]["lines"][0]["total"] == 2
    assert res["heatmap"]["values"] == live.heat.grid()["values"]
    csv_text = zn.events_to_csv(res["events"])
    assert csv_text.splitlines()[0].startswith("n,t,type,kind,id") and len(csv_text.splitlines()) == len(ev_live) + 1


def test_rest_demo(tmp_path):
    from fastapi.testclient import TestClient

    from peleng.config import load_config
    from peleng.server import create_app

    app = create_app(load_config(), demo=True, data_dir=tmp_path)
    with TestClient(app) as c:
        assert c.get("/api/zones/").json()["zones"] == []
        # человек 1 ходит по эллипсу 1.5×1.0 м с периодом 14 с — зона на пути вдоль x от 1.0 до 2.0 (правая половина эллипса)
        r = c.post("/api/zones/zone", json={"rect": [0.9, -1.5, 2.2, 1.5], "name": "Дверь"})
        assert r.status_code == 200, r.text
        zid = r.json()["id"]
        assert c.post("/api/zones/zone", json={"rect": [0, 0, 0, 0]}).status_code == 400
        assert c.post("/api/zones/line", json={"a": [0.0, -2.0], "b": [0.0, 2.0], "name": "Проход"}).status_code == 200
        assert (tmp_path / "zones.json").is_file()
        time.sleep(16.0)
        ev = c.get("/api/zones/events").json()
        assert ev["total"] >= 2, ev
        kinds = {e["type"] for e in ev["events"]}
        assert "enter" in kinds and "line_cross" in kinds
        st = c.get("/api/zones/stats").json()
        z = st["zones"][0]
        assert z["id"] == zid and z["visits"] >= 1
        assert st["lines"][0]["total"] >= 1
        hm = c.get("/api/zones/heatmap").json()
        assert hm["nx"] == 25 and hm["ny"] == 20 and hm["total_s"] > 5 and hm["max"] > 0
        assert c.get("/api/zones/events.csv").text.startswith("n,t,type")
        assert c.post("/api/zones/heatmap/reset").json()["ok"]
        assert c.get("/api/zones/heatmap").json()["total_s"] < 1.0
        assert c.put(f"/api/zones/zone/{zid}", json={"name": "Выход"}).json()["name"] == "Выход"
        # запись сессии: события лежат внутри frame, анализ по записи даёт события
        c.post("/api/record/start")
        time.sleep(8.0)
        name = Path(c.post("/api/record/stop").json()["file"]).name
        rec = [json.loads(l) for l in (tmp_path / "sessions" / name).read_text(encoding="utf-8").splitlines()]
        assert any("zones" in m and "inside" in m["zones"] for m in rec if m.get("type") == "frame")
        a = c.get("/api/zones/analyze", params={"session": name}).json()
        assert a["frames"] > 20 and a["summary"]["zones"][0]["name"] == "Выход" and a["heatmap"]["total_s"] > 3
        assert c.get("/api/zones/analyze.csv", params={"session": name}).status_code == 200
        assert c.get("/api/zones/analyze", params={"session": "../x.jsonl"}).status_code in (400, 404)
        assert c.get("/api/zones/analyze", params={"session": "nope.jsonl"}).status_code == 404
        assert c.delete(f"/api/zones/zone/{zid}").json()["ok"]
        assert c.delete(f"/api/zones/zone/{zid}").status_code == 404
