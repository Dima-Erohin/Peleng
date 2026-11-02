"""Расширение «Точность»: pos_std_m в сообщениях frame, проверки по доске и рулетке, отчёт.

REST (/api/accuracy/...):
  POST board            — проверка по доске (доска лежит где угодно, видна ≥ 2 камерам)
  POST point/a, point/b — запомнить точку: среднее положение человека/объекта за avg_s секунд ({"kind", "id"} — необязательно)
  POST tape             — {"measured_m": 2.5} → ошибка A→B в см и %
  GET  points           — текущие A и B
  GET  history, POST history/clear — история проверок (data/accuracy.json)
  GET  report           — отчёт: калибровка камер, углы между камерами, σ в центре зоны, сводка проверок
Подробно — docs/accuracy.md.
"""
from __future__ import annotations

import asyncio
import math
import time
from collections import deque

import numpy as np
from fastapi import HTTPException

from .. import accuracy as acc

ORDER = 80  # после расширения объектов (A): дополняем и msg["objects"]

FULL_RES_SWITCH_S = 0.5  # сколько ждать, пока телефоны переключатся на полное разрешение (как в server.py)

DEFAULTS = {
    "sigma_px": 2.0,  # шум положения точки скелета/объекта в кадре трекинга, пиксели
    "board_sigma_px": 0.5,  # шум углов доски (субпиксельные) — для «ожидаемой» ошибки проверки по доске
    "board_collect_s": 1.5,  # сколько секунд собирать кадры доски
    "board_reproj_px": 5.0,  # порог ошибки репроекции угла доски
    "avg_s": 1.0,  # усреднение положения для точек A/B
    "buffer_s": 10.0,  # сколько секунд положений помнить
    "zone_center": None,  # центр рабочей зоны [x, y, z]; по умолчанию — центр комнаты на высоте 0.9 м
}


def _key(kind: str, item: dict) -> str:
    return f"{kind}:{item.get('id')}"


def register(app, ctx) -> None:
    cfg = {**DEFAULTS, **(ctx.cfg.get("accuracy") or {})}
    hist_path = ctx.data_dir / "accuracy.json"
    buf: deque = deque()  # (t, [{"key", "pos", "floor", "std", "mode"}])
    st: dict = {"a": None, "b": None, "sizes": {}}

    # ------------------------------------------------------------------ хук кадра
    def accuracy_hook(fctx) -> None:
        msg = fctx.msg
        sizes = msg.get("frame_size") or {}
        st["sizes"] = sizes
        sigma = float(cfg["sigma_px"])

        def model(cid):
            m = fctx.cams.get(cid)
            if m is None:
                size = sizes.get(cid)
                m = ctx.registry.model(cid, tuple(size)) if size else ctx.registry.model(cid)
            return m if m is not None and m.calibrated else None

        samples = []
        for kind, field in (("person", "persons"), ("object", "objects")):
            for it in msg.get(field) or []:
                std = None
                if it.get("mode") == "tri" and it.get("pos") is not None:
                    models = [m for m in (model(c) for c in it.get("cams") or []) if m is not None]
                    s = acc.pos_std(models, np.asarray(it["pos"], float), sigma)
                    if s is not None:
                        std = [round(float(v), 4) for v in s]
                        it["pos_std_m"] = std
                if it.get("pos") is not None and not it.get("predicted"):
                    samples.append({"key": _key(it.get("kind", kind), it), "pos": it["pos"], "floor": it.get("floor"),
                                    "std": std, "mode": it.get("mode")})
        buf.append((fctx.t, samples))
        while buf and buf[0][0] < fctx.t - float(cfg["buffer_s"]):
            buf.popleft()

    ctx.pipeline.frame_hooks.append(accuracy_hook)
    ctx.pipeline.reset_hooks.append(lambda: buf.clear())

    # ------------------------------------------------------------------ проверка по доске
    def _collect_board(ids: list[str]) -> dict[str, list]:
        found: dict[str, list] = {c: [] for c in ids}
        seen: dict[str, int] = {}
        with ctx.pipeline.full_res(ids):
            t_end = time.time() + FULL_RES_SWITCH_S + float(cfg["board_collect_s"])
            while time.time() < t_end:
                for c in ids:
                    src = ctx.registry.sources.get(c)
                    f = src.latest() if src else None
                    if f is None or seen.get(c) == f.seq:
                        continue
                    seen[c] = f.seq
                    if not src.is_full_res(f):  # кадр, запрошенный ещё до переключения
                        continue
                    d = ctx.board.detect(f.image)
                    if d is not None:
                        found[c].append((f, d))
                time.sleep(0.03)
        return found

    def _do_board(cam_ids: list[str] | None) -> dict:
        srcs = ctx.registry.sources
        ids = [c for c in (cam_ids or list(srcs)) if c in srcs and srcs[c].online]
        if len(ids) < 2:
            return {"ok": False, "message": "нужно ≥ 2 активных камер"}
        found = _collect_board(ids)
        cams, dets, per_cam = {}, {}, {}
        for c in ids:
            items = found[c]
            if not items:
                per_cam[c] = "доска не видна"
                continue
            f0 = items[-1][0]
            h, w = f0.image.shape[:2]
            m = ctx.registry.model(c, (w, h))
            if m is None or not m.calibrated:
                per_cam[c] = "не откалибрована"
                continue
            cams[c] = m
            dets[c] = acc.merge_detections([d for _, d in items], ctx.board)
            per_cam[c] = f"{len(items)} кадров, {len(dets[c].ids)} углов"
        if not dets:
            return {"ok": False, "message": "доска не видна ни одной откалиброванной камере", "per_cam": per_cam}
        res = acc.board_check(ctx.board, cams, dets, float(cfg["board_reproj_px"]), float(cfg["board_sigma_px"]))
        res["per_cam"] = per_cam
        if res["ok"]:
            acc.append_history(hist_path, {"type": "board", **res})
        return res

    @app.post("/api/accuracy/board")
    async def api_board(body: dict | None = None):
        return await asyncio.to_thread(_do_board, (body or {}).get("cameras"))

    # ------------------------------------------------------------------ рулетка: точки A, B
    def _pick(t0: float, t1: float, want: str | None) -> tuple[str | None, list[dict]]:
        by_key: dict[str, list[dict]] = {}
        for t, samples in list(buf):
            if t0 <= t <= t1:
                for s in samples:
                    by_key.setdefault(s["key"], []).append(s)
        if not by_key:
            return None, []
        if want and want in by_key:
            return want, by_key[want]
        # иначе — тот, кого дольше всего видели по ≥ 2 камерам
        k = max(by_key, key=lambda k: (sum(s["mode"] == "tri" for s in by_key[k]), len(by_key[k])))
        return k, by_key[k]

    @app.post("/api/accuracy/point/{which}")
    async def api_point(which: str, body: dict | None = None):
        which = which.lower()
        if which not in ("a", "b"):
            raise HTTPException(404, "точка a или b")
        body = body or {}
        want = f"{body['kind']}:{body['id']}" if body.get("id") is not None else None
        if want is None and which == "b" and st["a"]:
            want = st["a"]["key"]  # по умолчанию — тот же человек/объект, что в точке A
        avg_s = float(body.get("avg_s", cfg["avg_s"]))
        t0 = time.time()
        await asyncio.sleep(avg_s + 0.35)  # +запас на обработку последних кадров
        key, samples = _pick(t0 - 0.05, t0 + avg_s, want)
        if not samples:  # конвейер мог отставать — берём последние avg_s секунд
            t_last = buf[-1][0] if buf else 0.0
            key, samples = _pick(t_last - avg_s, t_last, want)
        if not samples:
            raise HTTPException(409, "в кадре нет ни человека, ни объекта")
        p = acc.average_point(samples)
        p.update(key=key, t=round(t0, 2))
        if p["spread_m"] > 0.1:
            p["warning"] = f"двигался во время замера (разброс {p['spread_m'] * 100:.0f} см) — стойте неподвижно"
        st[which] = p
        if which == "a":
            st["b"] = None
        return p

    @app.get("/api/accuracy/points")
    async def api_points():
        return {"a": st["a"], "b": st["b"]}

    @app.post("/api/accuracy/tape")
    async def api_tape(body: dict):
        a, b = st["a"], st["b"]
        if not a or not b:
            raise HTTPException(409, "сначала отметьте точки A и B")
        try:
            measured = float(body.get("measured_m"))
        except (TypeError, ValueError):
            raise HTTPException(400, "measured_m — расстояние по рулетке, м") from None
        if not (measured > 0):
            raise HTTPException(400, "расстояние по рулетке должно быть > 0")
        horizontal = bool(body.get("horizontal", a["key"].startswith("person:")))
        res = acc.tape_result(a, b, measured, horizontal)
        res.update(type="tape", a=a, b=b, target=a["key"] if a["key"] == b["key"] else f"{a['key']} → {b['key']}")
        acc.append_history(hist_path, res)
        return res

    # ------------------------------------------------------------------ история и отчёт
    @app.get("/api/accuracy/history")
    async def api_history():
        return acc.load_history(hist_path)

    @app.post("/api/accuracy/history/clear")
    async def api_history_clear():
        if hist_path.exists():
            hist_path.unlink()
        return {"ok": True}

    def _zone_center() -> np.ndarray:
        if cfg.get("zone_center") is not None:
            return np.asarray(cfg["zone_center"], float)
        off = (ctx.cfg.get("room") or {}).get("board_offset_m") or [0.0, 0.0]
        return np.array([-float(off[0]), -float(off[1]), 0.9])

    def _report() -> dict:
        models = {}
        for cid in sorted(set(ctx.registry.sources) | set(ctx.registry.models)):
            m = ctx.registry.model(cid)
            if m is None:
                continue
            size = st["sizes"].get(cid)  # σ задана в пикселях кадра трекинга
            models[cid] = m.resized(tuple(size)) if size else m
        rep = acc.camera_report(models, _zone_center(), float(cfg["sigma_px"]))
        items = list(buf)
        recent = [s for t, ss in items for s in ss if s["std"] is not None and t >= items[-1][0] - 3.0] if items else []
        if recent:
            h = [math.hypot(s["std"][0], s["std"][1]) * 100 for s in recent]
            rep["live"] = {"samples": len(recent), "sigma_h_cm_median": round(float(np.median(h)), 1), "sigma_h_cm_max": round(float(np.max(h)), 1)}
        rep["checks"] = acc.summarize_history(acc.load_history(hist_path))
        return rep

    @app.get("/api/accuracy/report")
    async def api_report():
        return await asyncio.to_thread(_report)
