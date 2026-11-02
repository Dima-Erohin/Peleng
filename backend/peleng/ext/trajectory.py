"""Расширение «Траектории»: восстановление траекторий из записанных сессий и экспорт CSV/JSON.

REST (всё под /api/trajectory/):
  GET /sessions                                   — записи в data/sessions (*.jsonl): имя, размер, время
  GET /restore?session=…&max_gap_s=…&rate_hz=…    — восстановить: метрики треков + линии для 3D (прореженные)
  GET /export?session=…&fmt=csv|json&data=restored|raw&max_gap_s=…&rate_hz=…  — скачать файл

Настройки — cfg["trajectory"] (значения по умолчанию в peleng/trajectory.py: DEFAULTS).
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from .. import trajectory as tj

_NAME_RE = re.compile(r"^[\w.\-]+\.jsonl$")
_MAX_VIEW_POINTS = 3000  # на трек во вьювер — не больше стольких точек


def register(app: FastAPI, ctx) -> None:
    sessions_dir = Path(ctx.data_dir) / "sessions"
    cache: dict = {}  # (имя, mtime, размер, параметры) -> SessionResult; последние несколько
    lock = threading.Lock()

    def session_path(name: str) -> Path:
        if not name or not _NAME_RE.match(name):
            raise HTTPException(400, "недопустимое имя сессии")
        p = (sessions_dir / name).resolve()
        if p.parent != sessions_dir.resolve() or not p.is_file():
            raise HTTPException(404, f"нет такой сессии: {name}")
        return p

    def compute(name: str, max_gap_s: float | None, rate_hz: float | None) -> tj.SessionResult:
        p = session_path(name)
        try:
            s = tj.settings(ctx.cfg, max_gap_s=max_gap_s, rate_hz=rate_hz)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        st = p.stat()
        key = (name, st.st_mtime_ns, st.st_size, s["max_gap_s"], s["rate_hz"])
        with lock:
            if key in cache:
                return cache[key]
        res = tj.restore_session(p, ctx.cfg, max_gap_s=max_gap_s, rate_hz=rate_hz)
        with lock:
            cache[key] = res
            while len(cache) > 4:
                cache.pop(next(iter(cache)))
        return res

    @app.get("/api/trajectory/sessions")
    async def api_sessions():
        if not sessions_dir.is_dir():
            return []
        out = []
        for p in sessions_dir.glob("*.jsonl"):
            st = p.stat()
            out.append({"name": p.name, "size": st.st_size, "mtime": st.st_mtime,
                        "recording": ctx.pipeline.recording is not None and Path(ctx.pipeline.recording).name == p.name})
        out.sort(key=lambda x: x["mtime"], reverse=True)
        return out

    @app.get("/api/trajectory/restore")
    async def api_restore(session: str, max_gap_s: float | None = None, rate_hz: float | None = None):
        def work():
            res = compute(session, max_gap_s, rate_hz)
            out = tj.summary(res)
            out["session"] = session
            # линии для вьювера: восстановленные и сырые (только измерения), прореженные
            out["restored"] = [_view(r.kind, r.id, r.t, r.pos, extra={"source": r.source}) for r in res.restored]
            out["raw"] = [_view(tr.kind, tr.id, tr.t[tr.measured], tr.pos[tr.measured]) for tr in res.raw if tr.measured.any()]
            return out

        return await asyncio.to_thread(work)

    @app.get("/api/trajectory/export")
    async def api_export(session: str, fmt: str = "csv", data: str = "restored", max_gap_s: float | None = None, rate_hz: float | None = None):
        if fmt not in ("csv", "json") or data not in ("restored", "raw"):
            raise HTTPException(400, "fmt: csv|json, data: restored|raw")

        def work():
            res = compute(session, max_gap_s, rate_hz)
            raw = data == "raw"
            if fmt == "csv":
                return tj.to_csv(res, raw=raw)
            j = tj.to_json(res, raw=raw)
            j["session"] = session
            return json.dumps(j, ensure_ascii=False)

        body = await asyncio.to_thread(work)
        fname = f"{Path(session).stem}_{data}.{fmt}"
        media = "text/csv; charset=utf-8" if fmt == "csv" else "application/json"
        return Response(body, media_type=media, headers={"Content-Disposition": f'attachment; filename="{fname}"'})


def _view(kind: str, tid: int, t, pos, extra: dict | None = None) -> dict:
    step = max(1, -(-len(t) // _MAX_VIEW_POINTS))
    out = {"kind": kind, "id": tid, "t": [round(float(x), 3) for x in t[::step]], "pos": [[round(float(v), 3) for v in p] for p in pos[::step]]}
    for k, v in (extra or {}).items():
        out[k] = v[::step]
    return out
