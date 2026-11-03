"""Расширение «Зоны и события»: прямоугольники/многоугольники на полу, линии подсчёта, тепловая карта.

В сообщение "frame" добавляется поле "zones": {"inside": {id_зоны: [кто внутри]}, "lines": {id: счётчики}, "events": [новые события]}.
Зоны хранятся в data/zones.json. Логика — peleng/zones.py, описание и формат — docs/zones.md.

REST (всё под /api/zones/):
  GET    /                         — зоны, линии, параметры
  POST   /zone, PUT/DELETE /zone/{id}   — зоны ({"rect":[x1,y1,x2,y2]} или {"points":[[x,y],…]}, name, color)
  POST   /line, PUT/DELETE /line/{id}   — линии подсчёта ({"a":[x,y],"b":[x,y]}, name, color)
  GET    /events?limit=100&after=n — журнал событий (в памяти); GET /events.csv — CSV; POST /reset — обнулить счётчики и журнал
  GET    /stats                    — сводка: сейчас внутри, посещений, время; по линиям — A→B / B→A
  GET    /heatmap                  — тепловая карта (live); POST /heatmap/reset — сбросить
  GET    /sessions                 — записи в data/sessions
  GET    /analyze?session=…&cell_m=… — прогнать запись через те же правила: события + сводка + тепловая карта; /analyze.csv — события CSV
"""
from __future__ import annotations

import asyncio
import re
import threading
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import Response

from .. import zones as zn

ORDER = 70  # после расширения объектов: учитываем и msg["objects"]
_NAME_RE = re.compile(r"^[\w.\-]+\.jsonl$")


def register(app, ctx) -> None:
    params = zn.settings(ctx.cfg)
    sessions_dir = Path(ctx.data_dir) / "sessions"
    store = zn.ZoneStore(Path(ctx.data_dir) / "zones.json")
    lock = threading.RLock()
    eng = zn.Engine(store.zones, store.lines, params, room=ctx.cfg.get("room"))
    last_t = [0.0]

    def frame_hook(fc) -> None:
        with lock:
            events = eng.process(fc.msg)
            last_t[0] = fc.t
            if eng.zones or eng.lines or events:
                fc.msg["zones"] = eng.snapshot(fc.t, events)

    def reset_hook() -> None:
        with lock:
            eng.close_all("reset")  # события закрытия остаются в журнале; вьювер увидит их по /events

    ctx.pipeline.frame_hooks.append(frame_hook)
    ctx.pipeline.reset_hooks.append(reset_hook)

    def changed() -> dict:
        """Зоны изменились: обновить движок и сообщить вьюверам."""
        with lock:
            eng.set_geometry(store.zones, store.lines)
        snap = store.snapshot()
        ctx.hub.publish({"type": "zones", **snap})
        return snap

    def bad(e: Exception):
        raise HTTPException(400, str(e)) from e

    @app.get("/api/zones/")
    async def api_all():
        return {**store.snapshot(), "params": params}

    @app.post("/api/zones/zone")
    async def api_zone_add(body: dict):
        try:
            z = store.add_zone(body)
        except ValueError as e:
            bad(e)
        changed()
        return z

    @app.put("/api/zones/zone/{zid}")
    async def api_zone_put(zid: str, body: dict):
        try:
            z = store.update("zone", zid, body)
        except KeyError:
            raise HTTPException(404, f"нет зоны {zid}") from None
        except ValueError as e:
            bad(e)
        changed()
        return z

    @app.delete("/api/zones/zone/{zid}")
    async def api_zone_del(zid: str):
        if not store.delete("zone", zid):
            raise HTTPException(404, f"нет зоны {zid}")
        changed()
        return {"ok": True}

    @app.post("/api/zones/line")
    async def api_line_add(body: dict):
        try:
            l = store.add_line(body)
        except ValueError as e:
            bad(e)
        changed()
        return l

    @app.put("/api/zones/line/{lid}")
    async def api_line_put(lid: str, body: dict):
        try:
            l = store.update("line", lid, body)
        except KeyError:
            raise HTTPException(404, f"нет линии {lid}") from None
        except ValueError as e:
            bad(e)
        changed()
        return l

    @app.delete("/api/zones/line/{lid}")
    async def api_line_del(lid: str):
        if not store.delete("line", lid):
            raise HTTPException(404, f"нет линии {lid}")
        changed()
        return {"ok": True}

    @app.get("/api/zones/events")
    async def api_events(limit: int = 100, after: int = 0):
        with lock:
            ev = [e for e in eng.log if e["n"] > after]
            return {"events": ev[-max(1, min(limit, 2000)):], "total": eng.n_events}

    @app.get("/api/zones/events.csv")
    async def api_events_csv():
        with lock:
            body = zn.events_to_csv(list(eng.log))
        return Response(body, media_type="text/csv; charset=utf-8", headers={"Content-Disposition": 'attachment; filename="zone_events.csv"'})

    @app.post("/api/zones/reset")
    async def api_reset():
        with lock:
            eng.reset_stats()
        return {"ok": True}

    @app.get("/api/zones/stats")
    async def api_stats():
        with lock:
            return eng.stats(last_t[0])

    @app.get("/api/zones/heatmap")
    async def api_heat():
        with lock:
            return eng.heat.grid()

    @app.post("/api/zones/heatmap/reset")
    async def api_heat_reset():
        with lock:
            eng.heat.reset()
        return {"ok": True}

    @app.get("/api/zones/sessions")
    async def api_sessions():
        if not sessions_dir.is_dir():
            return []
        out = [{"name": p.name, "size": p.stat().st_size, "mtime": p.stat().st_mtime} for p in sessions_dir.glob("*.jsonl")]
        out.sort(key=lambda x: x["mtime"], reverse=True)
        return out

    def run_analysis(session: str, cell_m: float | None) -> dict:
        if not session or not _NAME_RE.match(session):
            raise HTTPException(400, "недопустимое имя сессии")
        p = (sessions_dir / session).resolve()
        if p.parent != sessions_dir.resolve() or not p.is_file():
            raise HTTPException(404, f"нет такой сессии: {session}")
        if cell_m is not None and not (0.02 <= cell_m <= 5):
            raise HTTPException(400, "cell_m — шаг сетки, м (0.02…5)")
        snap = store.snapshot()
        res = zn.analyze_session(p, snap["zones"], snap["lines"], params, cell_m)
        res["session"] = session
        return res

    @app.get("/api/zones/analyze")
    async def api_analyze(session: str, cell_m: float | None = None):
        res = await asyncio.to_thread(run_analysis, session, cell_m)
        res["events_total"] = len(res["events"])
        res["events"] = res["events"][:5000]  # для интерфейса; полный список — CSV
        return res

    @app.get("/api/zones/analyze.csv")
    async def api_analyze_csv(session: str):
        res = await asyncio.to_thread(run_analysis, session, None)
        return Response(zn.events_to_csv(res["events"]), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{Path(session).stem}_zone_events.csv"'})
