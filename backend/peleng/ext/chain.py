"""Расширение «Калибровка цепочкой»: положение камер, когда доску не видят все камеры сразу.

Процедура: «Начать» → снимок доски на полу (задаёт мир, видна ≥ 1 камере) → снимки доски, которую одновременно
видят ≥ 2 камеры (в руках, наклонно, вертикально) → «Применить». Математика — peleng/chain.py, описание — docs/chain-calibration.md.

REST (/api/chain/...):
  GET  /api/chain                    — состояние: снимки, граф связей камер, подсказки, последний расчёт
  POST /api/chain/start              — начать заново (прежние снимки забываются)
  POST /api/chain/snapshot           — снимок: ~1 с кадров в полном разрешении со всех онлайн-камер;
                                       {"floor": true} — доска снова лежит на полу (в другом месте): уточняет наклон пола
  POST /api/chain/snapshot/{k}/delete — убрать снимок k (опорный снимок 0 не удаляется — начните заново)
  POST /api/chain/auto               — {"on": true} автоснимок: доску видят ≥ 2 камеры, она неподвижна и сдвинута с прошлого раза
  POST /api/chain/solve              — рассчитать без сохранения (просмотр ошибок и предупреждений)
  POST /api/chain/apply              — рассчитать и сохранить положения связанных камер
  POST /api/chain/cancel             — отменить, ничего не меняя
Изменения состояния рассылаются вьюверам сообщением {"type": "chain", ...} (то же, что GET /api/chain).
"""
from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
from fastapi import HTTPException

from .. import chain as ch
from ..accuracy import merge_detections
from ..geometry import CameraModel

DEFAULTS = {
    "collect_s": 1.0,  # сколько секунд собирать кадры на один снимок
    "move_px": 2.5,  # доска «двигалась», если углы гуляли сильнее (px, в кадре полного разрешения)
    "min_corners": 6,
    "outlier_px": 3.0,  # наблюдение с ошибкой репроекции больше (и > 4× медианы) отбрасывается
    "warn_reproj_px": 1.0,
    "warn_std_m": 0.03,
    "warn_hops": 3,
    "auto_min_shift": 0.05,  # автоснимок: доска сдвинулась хотя бы на эту долю ширины кадра с прошлого снимка
}


def register(app, ctx) -> None:
    cfg = {**DEFAULTS, **(ctx.cfg.get("chain") or {})}
    lock = threading.Lock()  # один снимок за раз (ручной или авто)
    st: dict = {"active": False, "auto": False, "busy": False, "snaps": [], "obs": [], "next_k": 0, "result": None, "applied": None}

    def _collect(ids: list[str]) -> dict[str, list]:
        collect = getattr(app.state, "collect_board", None)
        if collect is None:
            raise HTTPException(500, "сервер не даёт сбор кадров доски (app.state.collect_board)")
        return collect(ids, float(cfg["collect_s"]))

    def _online() -> list[str]:
        return [c for c, s in list(ctx.registry.sources.items()) if s.online]

    def _prior_centers() -> dict[str, np.ndarray]:
        out = {}
        res = st["result"]
        for c in list(ctx.registry.models):
            m = ctx.registry.model(c)
            if m is not None and m.calibrated:
                out[c] = m.center
        if res and res.get("cams"):  # свежий расчёт точнее прежней калибровки
            for c, x in res["cams"].items():
                out[c] = np.asarray(x["center"], float)
        return out

    def status() -> dict:
        cams = sorted(set(_online()) | {o.cam for o in st["obs"]})
        g = ch.graph_summary(st["obs"], cams, anchor=0, prior_centers=_prior_centers())
        if not st["active"]:
            g["hints"] = ["Нажмите «Начать», затем положите доску на пол (видна хотя бы одной камере) и сделайте первый снимок."]
        return {"type": "chain", "active": st["active"], "auto": st["auto"], "busy": st["busy"], "snaps": st["snaps"],
                "graph": g, "result": st["result"], "applied": st["applied"]}

    def publish() -> None:
        ctx.hub.publish(status())

    # ------------------------------------------------------------------ снимок
    def _detections(ids: list[str]) -> tuple[dict, dict]:
        """Собрать кадры, по каждой камере — медиана углов неподвижной доски. -> (наблюдения c -> (Observation, spread), причины пропуска)"""
        found = _collect(ids)
        good, skipped = {}, {}
        for c in ids:
            items = found.get(c) or []
            if not items:
                continue
            f0 = items[-1][0]
            h, w = f0.image.shape[:2]
            same = [d for f, d in items if f.image.shape[:2] == (h, w)]
            d = merge_detections(same, ctx.board)
            if d is None or len(d.ids) < int(cfg["min_corners"]):
                skipped[c] = f"мало углов ({0 if d is None else len(d.ids)})"
                continue
            # неподвижность: насколько углы в отдельных кадрах отходят от медианы
            spread = 0.0
            pos = {int(i): k for k, i in enumerate(d.ids)}
            for x in same:
                dev = [np.linalg.norm(cc - d.corners[pos[int(i)]]) for i, cc in zip(x.ids, x.corners) if int(i) in pos]
                if dev:
                    spread = max(spread, float(np.mean(dev)))
            if spread > float(cfg["move_px"]):
                skipped[c] = f"доска двигалась ({spread:.1f} px) — держите неподвижно"
                continue
            m = ctx.registry.model(c, (w, h))
            obs = ch.Observation(-1, c, d.corners, d.ids, m.K.copy(), m.dist.copy(), (w, h))
            if not ch.observation_ok(obs, ctx.board):
                skipped[c] = "углы на одной линии"
                continue
            good[c] = (obs, spread, len(same))
        return good, skipped

    def _add(good: dict, skipped: dict, floor: bool) -> dict:
        k = st["next_k"]
        st["next_k"] += 1
        for obs, _, _ in good.values():
            obs.snap = k
            st["obs"].append(obs)
        snap = {"k": k, "time": round(time.time(), 2), "anchor": k == 0, "floor": floor or k == 0,
                "cams": {c: {"corners": len(o.ids), "frames": n, "spread_px": round(s, 2), "size": list(o.size)} for c, (o, s, n) in sorted(good.items())},
                "skipped": skipped}
        st["snaps"].append(snap)
        st["result"] = None
        return snap

    def _snapshot(auto: bool = False, floor: bool = False) -> dict:
        if not st["active"]:
            return {"ok": False, "message": "сначала «Начать»"}
        if not lock.acquire(blocking=not auto):
            return {"ok": False, "message": "снимок уже идёт"}
        changed = not auto
        try:
            st["busy"] = not auto
            ids = _online()
            if not ids:
                return {"ok": False, "message": "нет активных камер"}
            good, skipped = _detections(ids)
            first = not st["snaps"]
            need = 1 if first or floor else 2  # доску на полу достаточно видеть одной (уже привязанной) камере
            if len(good) < need:
                seen = ", ".join(good) or "ни одна камера"
                msg = "доска не видна" if not good else f"доску видит только {seen} — нужно ≥ 2 камеры одновременно"
                return {"ok": False, "message": msg, "seen": sorted(good), "skipped": skipped}
            if auto and not _auto_worth(good):
                return {"ok": False, "message": "та же поза или нет привязанной камеры", "seen": sorted(good)}
            snap = _add(good, skipped, floor)
            changed = True
            msg = "опорный снимок (доска на полу) — система координат задана" if first else \
                f"снимок {snap['k']}{' (на полу)' if floor else ''}: {', '.join(snap['cams'])}"
            if not first:
                g = ch.graph_summary(st["obs"], anchor=0)
                if not any(g["cams"][c]["linked"] for c in snap["cams"]):
                    msg += " (пока не связан с опорным снимком)"
            return {"ok": True, "message": msg, "snapshot": snap}
        finally:
            st["busy"] = False
            lock.release()
            if changed:  # автоснимок без результата не рассылаем — он пробует каждые ~1.5 с
                publish()

    def _auto_worth(good: dict) -> bool:
        """Автоснимок полезен: среди камер есть привязанная, и доска сдвинулась с прошлого снимка."""
        g = ch.graph_summary(st["obs"], anchor=0)
        if not any(g["cams"].get(c, {}).get("linked") for c in good):
            return False
        last_k = st["snaps"][-1]["k"]
        last = {o.cam: o for o in st["obs"] if o.snap == last_k}
        if set(last) != set(good):
            return True
        for c, (o, _, _) in good.items():
            p = last[c]
            common, ia, ib = np.intersect1d(o.ids, p.ids, return_indices=True)
            if len(common) < 4:
                return True
            if np.linalg.norm(o.corners[ia] - p.corners[ib], axis=1).mean() > float(cfg["auto_min_shift"]) * o.size[0]:
                return True
        return False

    def _auto_loop() -> None:
        while st["auto"] and st["active"]:
            if st["snaps"]:  # опорный снимок (на полу) — только вручную
                try:
                    r = _snapshot(auto=True)
                    if r.get("ok"):
                        print(f"[chain] автоснимок: {r['message']}")
                        time.sleep(1.0)  # дать перенести доску
                except Exception as e:  # noqa: BLE001
                    print(f"[chain] автоснимок: ошибка {e!r}")
                    time.sleep(1.0)
            time.sleep(0.3)

    # ------------------------------------------------------------------ расчёт и применение
    def _solve() -> ch.ChainResult:
        floor = {s["k"] for s in st["snaps"] if s.get("floor") and s["k"] != 0}
        res = ch.solve(ctx.board, st["obs"], anchor=0, floor_snaps=floor, outlier_px=float(cfg["outlier_px"]), warn_reproj_px=float(cfg["warn_reproj_px"]),
                       warn_std_m=float(cfg["warn_std_m"]), warn_hops=int(cfg["warn_hops"]))
        for c in res.cams:
            m = ctx.registry.model(c)
            if m is not None and m.meta.get("intrinsics", "approx") not in ("board", "demo"):
                res.warnings.append(f"{c}: оптика приблизительная (по углу обзора) — в цепочке ошибка накапливается; откалибруйте оптику")
        st["result"] = res.summary()
        return res

    def _do_solve() -> dict:
        if not st["obs"]:
            return {"ok": False, "message": "снимков нет"}
        _solve()
        publish()
        return st["result"]

    def _do_apply() -> dict:
        if not st["obs"]:
            return {"ok": False, "message": "снимков нет"}
        res = _solve()
        if not res.ok:
            publish()
            return st["result"]
        save = ctx.state.get("mode") == "live"  # демо/запись не трогают data/calib
        last_size = {o.cam: tuple(o.size) for o in st["obs"]}
        out = {}
        for c, x in res.cams.items():
            m = ctx.registry.model(c, last_size[c])
            meta = {k: v for k, v in m.meta.items() if not k.startswith("extr")}
            meta.update({
                "extrinsics_method": "chain",
                "extrinsics_reproj_px": round(float(x["reproj_px"]), 2),
                "extrinsics_time": time.time(),
                "extrinsics_points": int(x["points"]),
                "extrinsics_chain": {
                    "snapshots": x["snapshots"], "hops": x["hops"],
                    "center_std_m": None if x["center_std_m"] is None else np.round(x["center_std_m"], 4).tolist(),
                },
            })
            new = CameraModel(m.K, m.dist, m.size, x["R"], x["t"], meta)
            ctx.registry.set_model(c, new, save=save)
            out[c] = {"ok": True, **res.summary()["cams"][c], "dist_to_origin_m": round(float(np.linalg.norm(new.center)), 2)}
        for c in res.unlinked:
            out[c] = {"ok": False, "message": "не связана — положение не менялось"}
        ctx.pipeline.reset_tracking()
        ctx.hub.publish({"type": "cameras", "cameras": ctx.registry.describe()})
        st["applied"] = {"time": round(time.time(), 2), "cameras": sorted(res.cams), "saved": save, "reproj_px": st["result"]["reproj_px"]}
        st.update(active=False, auto=False)
        publish()
        return {**st["result"], "cameras": out, "saved": save}

    # ------------------------------------------------------------------ REST
    @app.get("/api/chain")
    async def api_state():
        return status()

    @app.post("/api/chain/start")
    async def api_start():
        with lock:
            st.update(active=True, auto=False, snaps=[], obs=[], next_k=0, result=None, applied=None)
        publish()
        return status()

    @app.post("/api/chain/cancel")
    async def api_cancel():
        st.update(auto=False)
        with lock:
            st.update(active=False, snaps=[], obs=[], next_k=0, result=None)
        publish()
        return status()

    @app.post("/api/chain/snapshot")
    async def api_snapshot(body: dict | None = None):
        return await asyncio.to_thread(_snapshot, False, bool((body or {}).get("floor")))

    @app.post("/api/chain/snapshot/{k}/delete")
    async def api_delete(k: int):
        if k == 0:
            raise HTTPException(400, "опорный снимок не удаляется — начните заново")
        with lock:
            if not any(s["k"] == k for s in st["snaps"]):
                raise HTTPException(404, f"нет снимка {k}")
            st["snaps"] = [s for s in st["snaps"] if s["k"] != k]
            st["obs"] = [o for o in st["obs"] if o.snap != k]
            st["result"] = None
        publish()
        return status()

    @app.post("/api/chain/auto")
    async def api_auto(body: dict | None = None):
        on = bool((body or {}).get("on", True))
        if on and not st["active"]:
            raise HTTPException(409, "сначала «Начать»")
        was = st["auto"]
        st["auto"] = on
        if on and not was:
            threading.Thread(target=_auto_loop, daemon=True, name="chain-auto").start()
        publish()
        return status()

    @app.post("/api/chain/solve")
    async def api_solve():
        return await asyncio.to_thread(_do_solve)

    @app.post("/api/chain/apply")
    async def api_apply():
        return await asyncio.to_thread(_do_apply)
