"""Расширение «Карта покрытия»: какие участки пола видят ≥ 2 / 1 / 0 камер (помощь при расстановке камер).

REST:
  GET /api/coverage?cell_m=0.2&height_m=0.9 — карта по откалиброванным камерам (формат — docs/coverage.md)

Сервер не рассылает карту сам: вьювер пересчитывает её по кнопке и по сообщению "cameras",
которое сервер и так шлёт после каждой калибровки.
Логика — peleng/coverage.py, настройки — cfg["coverage"] (см. coverage.DEFAULTS).
"""
from __future__ import annotations

import asyncio

from fastapi import HTTPException

from .. import coverage as cov


def register(app, ctx) -> None:
    def _params(cell_m: float | None, height_m: float | None) -> dict:
        p = {**cov.DEFAULTS, **(ctx.cfg.get("coverage") or {})}
        if p.get("ref_width_px") is None:
            p["ref_width_px"] = (ctx.cfg.get("pipeline") or {}).get("track_width", 640)
        if cell_m is not None:
            p["cell_m"] = cell_m
        if height_m is not None:
            p["height_m"] = height_m
        return p

    def _compute(params: dict) -> dict:
        models = {}
        for cid in sorted(set(ctx.registry.sources) | set(ctx.registry.models)):
            m = ctx.registry.model(cid)  # модель под любое разрешение: видимость считается в долях кадра
            if m is not None and m.calibrated:
                models[cid] = m
        return cov.coverage_map(models, ctx.cfg.get("room"), params)

    @app.get("/api/coverage")
    async def api_coverage(cell_m: float | None = None, height_m: float | None = None):
        if cell_m is not None and not (0 < cell_m <= 5):
            raise HTTPException(400, "cell_m — шаг сетки, м (0 < cell_m ≤ 5)")
        if height_m is not None and not (-1 <= height_m <= 5):
            raise HTTPException(400, "height_m — высота плоскости, м")
        return await asyncio.to_thread(_compute, _params(cell_m, height_m))
