"""Зоны и события: прямоугольники/многоугольники на полу, линии подсчёта, тепловая карта присутствия.

Логика без FastAPI и без камер: на вход сообщения "frame" (persons/objects с id, floor, predicted), на выход события.
Один и тот же Engine работает вживую (хук конвейера) и офлайн (прогон записи .jsonl) — события совпадают.

Правила событий (подробно — docs/zones.md):
  * положение — точка `floor` трека (метры, Z вверх); ключ трека (kind, id), kind = "person" или kind объекта;
  * расстояние до границы зоны со знаком: «+» внутри, «−» снаружи. Гистерезис hysteresis_m = ширина полосы вокруг границы:
    вход засчитывается при d ≥ +h/2, выход — при d ≤ −h/2; внутри полосы состояние не меняется;
  * плюс выдержка min_time_s: условие должно держаться непрерывно столько секунд. Время события — момент первого кадра
    выдержки, поэтому dwell (время в зоне) не искажается задержкой подтверждения;
  * predicted-позиции (трек не виден, положение — прогноз фильтра) в событиях не участвуют вообще: вход по прогнозу
    не начинается, выход по прогнозу не подтверждается, таймеры выдержки на них не идут;
  * трек пропал из persons/objects дольше lost_after_s → выход из всех зон (reason="lost") в момент последнего
    измеренного положения; сброс треков (перекалибровка) — reason="reset"; конец записи — reason="end";
  * линия подсчёта: сторона определяется знаком расстояния до прямой, учитывается с отступом h/2 (шум у линии
    не считается). Смена стороны между соседними «уверенными» положениями = пересечение, если соединяющий их
    отрезок пересекает отрезок линии (мимо концов — не считается). A→B: с левой стороны вектора A→B на правую.
"""
from __future__ import annotations

import json
import math
import re
import threading
from collections import deque
from pathlib import Path

DEFAULTS: dict = {
    "hysteresis_m": 0.15,  # ширина полосы у границы зоны/линии (±h/2 от границы)
    "min_time_s": 0.3,  # выдержка подтверждения входа/выхода
    "lost_after_s": 1.0,  # трек не виден в сообщениях дольше — считаем пропавшим
    "cell_m": 0.2,  # шаг сетки тепловой карты
    "max_extent_m": 30.0,  # без размера комнаты карта растёт динамически, но не шире этого (вокруг начала координат)
    "max_dt_s": 0.5,  # один шаг тепловой карты не добавляет больше секунд (обрывы потока)
    "events_max": 2000,  # размер журнала событий в памяти
}

PALETTE = ["#58a6ff", "#f0883e", "#3fb950", "#bc8cff", "#f778ba", "#d29922", "#39c5cf", "#ff7b72"]
_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def settings(cfg: dict | None = None) -> dict:
    s = {**DEFAULTS, **((cfg or {}).get("zones", {}) or {})}
    for k in DEFAULTS:
        s[k] = float(s[k]) if k != "events_max" else int(s[k])
    return s


# ---------------------------------------------------------------- геометрия
def poly_signed_dist(poly: list, x: float, y: float) -> float:
    """Расстояние до границы многоугольника, м: «+» внутри, «−» снаружи."""
    inside = False
    best = math.inf
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        u = 0.0 if L2 == 0 else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / L2))
        best = min(best, math.hypot(x - (x1 + u * dx), y - (y1 + u * dy)))
    return best if inside else -best


def line_side(a: list, b: list, x: float, y: float) -> float:
    """Расстояние до прямой AB со знаком: «−» слева от вектора A→B (сторона A), «+» справа (сторона B)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    return ((x - a[0]) * dy - (y - a[1]) * dx) / L


def segment_param(a: list, b: list, p: tuple, q: tuple) -> float | None:
    """Параметр t∈[0,1] вдоль AB, где отрезок PQ пересекает AB; None — не пересекает."""
    rx, ry = b[0] - a[0], b[1] - a[1]
    sx, sy = q[0] - p[0], q[1] - p[1]
    den = rx * sy - ry * sx
    if abs(den) < 1e-12:
        return None
    px, py = p[0] - a[0], p[1] - a[1]
    t = (px * sy - py * sx) / den
    u = (px * ry - py * rx) / den
    return t if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0 else None


# ---------------------------------------------------------------- хранение
class ZoneStore:
    """Зоны и линии в JSON-файле (data/zones.json): переживают перезапуск. Потокобезопасно."""

    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self.lock = threading.RLock()
        self.zones: list[dict] = []
        self.lines: list[dict] = []
        self._load()

    def _load(self) -> None:
        if not self.path or not self.path.is_file():
            return
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            self.zones = [self._norm_zone(z, keep_id=True) for z in d.get("zones", [])]
            self.lines = [self._norm_line(l, keep_id=True) for l in d.get("lines", [])]
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"[zones] не удалось прочитать {self.path}: {e!r}")
            self.zones, self.lines = [], []

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"zones": self.zones, "lines": self.lines}, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def _next_id(self, prefix: str, items: list[dict]) -> str:
        nums = [int(i["id"][1:]) for i in items if re.fullmatch(prefix + r"\d+", i["id"])]
        return f"{prefix}{max(nums, default=0) + 1}"

    # --- нормализация/проверка ---
    @staticmethod
    def _color(c, default: str) -> str:
        if c is None:
            return default
        if not isinstance(c, str) or not _COLOR_RE.match(c):
            raise ValueError("color — строка вида #rrggbb")
        return c.lower()

    @staticmethod
    def _xy(p, what: str) -> list[float]:
        if not (isinstance(p, (list, tuple)) and len(p) >= 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p[:2])):
            raise ValueError(f"{what}: нужна точка [x, y] в метрах")
        return [round(float(p[0]), 3), round(float(p[1]), 3)]

    def _norm_zone(self, d: dict, keep_id: bool = False, base: dict | None = None) -> dict:
        d = {**(base or {}), **d}
        if d.get("rect") is not None:
            r = d["rect"]
            if not (isinstance(r, (list, tuple)) and len(r) == 4):
                raise ValueError("rect — [x1, y1, x2, y2]")
            (x1, y1), (x2, y2) = self._xy(r[:2], "rect"), self._xy(r[2:], "rect")
            x1, x2, y1, y2 = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
            pts, typ = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], "rect"
        elif d.get("points") is not None:
            pts = [self._xy(p, "points") for p in d["points"]]
            typ = d.get("type") if d.get("type") in ("rect", "poly") else "poly"
            if typ == "rect" and len(pts) != 4:
                typ = "poly"
        else:
            raise ValueError("нужен rect [x1,y1,x2,y2] или points [[x,y], ...]")
        if len(pts) < 3:
            raise ValueError("многоугольнику нужно ≥ 3 точек")
        area = abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))) / 2
        if area < 0.01:
            raise ValueError("зона слишком мала (< 0.01 м²)")
        out = {"id": d.get("id"), "name": str(d.get("name") or "").strip()[:60], "color": d.get("color"), "type": typ, "points": pts}
        if not keep_id:
            out["id"] = (base or {}).get("id")
        return out

    def _norm_line(self, d: dict, keep_id: bool = False, base: dict | None = None) -> dict:
        d = {**(base or {}), **d}
        a, b = self._xy(d.get("a"), "a"), self._xy(d.get("b"), "b")
        if math.hypot(b[0] - a[0], b[1] - a[1]) < 0.1:
            raise ValueError("линия слишком короткая (< 0.1 м)")
        return {"id": d.get("id"), "name": str(d.get("name") or "").strip()[:60], "color": d.get("color"), "a": a, "b": b}

    # --- операции ---
    def add_zone(self, d: dict) -> dict:
        with self.lock:
            z = self._norm_zone(d)
            z["id"] = self._next_id("z", self.zones)
            z["name"] = z["name"] or f"Зона {z['id'][1:]}"
            z["color"] = self._color(d.get("color"), PALETTE[(int(z["id"][1:]) - 1) % len(PALETTE)])
            self.zones.append(z)
            self._save()
            return z

    def add_line(self, d: dict) -> dict:
        with self.lock:
            l = self._norm_line(d)
            l["id"] = self._next_id("l", self.lines)
            l["name"] = l["name"] or f"Линия {l['id'][1:]}"
            l["color"] = self._color(d.get("color"), PALETTE[(int(l["id"][1:]) + 3) % len(PALETTE)])
            self.lines.append(l)
            self._save()
            return l

    def update(self, kind: str, item_id: str, d: dict) -> dict:
        with self.lock:
            lst = self.zones if kind == "zone" else self.lines
            for i, old in enumerate(lst):
                if old["id"] == item_id:
                    if kind == "zone" and ("rect" in d or "points" in d):
                        old = {k: v for k, v in old.items() if k not in ("points", "type")}  # новая геометрия вместо старой
                    new = self._norm_zone(d, base=old) if kind == "zone" else self._norm_line(d, base=old)
                    new["id"] = item_id
                    new["name"] = new["name"] or lst[i]["name"]
                    new["color"] = self._color(d.get("color"), lst[i]["color"])
                    lst[i] = new
                    self._save()
                    return new
            raise KeyError(item_id)

    def delete(self, kind: str, item_id: str) -> bool:
        with self.lock:
            lst = self.zones if kind == "zone" else self.lines
            n = len(lst)
            lst[:] = [x for x in lst if x["id"] != item_id]
            if len(lst) != n:
                self._save()
            return len(lst) != n

    def snapshot(self) -> dict:
        with self.lock:
            return {"zones": json.loads(json.dumps(self.zones)), "lines": json.loads(json.dumps(self.lines))}


# ---------------------------------------------------------------- тепловая карта
def room_bounds(room: dict | None) -> tuple[list[float], list[float]] | None:
    """Границы пола по room.size_m (центр комнаты — минус board_offset_m, как в карте покрытия) или None."""
    if not room or not room.get("size_m"):
        return None
    w, d = float(room["size_m"][0]), float(room["size_m"][1])
    off = room.get("board_offset_m") or [0.0, 0.0]
    cx, cy = -float(off[0]), -float(off[1])
    return [cx - w / 2, cy - d / 2], [cx + w / 2, cy + d / 2]


class HeatMap:
    """Время присутствия (с) по сетке пола. Размер комнаты известен — сетка по ней, иначе растёт по мере надобности."""

    def __init__(self, cell_m: float = 0.2, room: dict | None = None, max_extent_m: float = 30.0):
        self.cell = float(cell_m)
        self.bounds = room_bounds(room)
        self.half = max_extent_m / 2
        self.cells: dict[tuple[int, int], float] = {}
        self.total = 0.0
        if self.bounds:
            lo, hi = self.bounds
            self.nx = max(1, math.ceil((hi[0] - lo[0]) / self.cell - 1e-9))
            self.ny = max(1, math.ceil((hi[1] - lo[1]) / self.cell - 1e-9))

    def reset(self) -> None:
        self.cells.clear()
        self.total = 0.0

    def add(self, x: float, y: float, dt: float) -> None:
        if dt <= 0 or not (math.isfinite(x) and math.isfinite(y)):
            return
        if self.bounds:
            ix, iy = math.floor((x - self.bounds[0][0]) / self.cell), math.floor((y - self.bounds[0][1]) / self.cell)
            if not (0 <= ix < self.nx and 0 <= iy < self.ny):
                return
        else:
            if abs(x) > self.half or abs(y) > self.half:
                return
            ix, iy = math.floor(x / self.cell), math.floor(y / self.cell)
        self.cells[(ix, iy)] = self.cells.get((ix, iy), 0.0) + dt
        self.total += dt

    def grid(self) -> dict:
        """origin — угол клетки (0,0), values — построчно (по y от меньшего к большему, внутри строки по x), секунды."""
        if self.bounds:
            ox, oy, nx, ny = self.bounds[0][0], self.bounds[0][1], self.nx, self.ny
            shift = (0, 0)
        elif self.cells:
            xs = [k[0] for k in self.cells]
            ys = [k[1] for k in self.cells]
            shift = (min(xs), min(ys))
            nx, ny = max(xs) - shift[0] + 1, max(ys) - shift[1] + 1
            ox, oy = shift[0] * self.cell, shift[1] * self.cell
        else:
            return {"origin": [0.0, 0.0], "cell_m": self.cell, "nx": 0, "ny": 0, "values": [], "max": 0.0, "total_s": 0.0, "bounds_from": "dynamic"}
        vals = [0.0] * (nx * ny)
        for (ix, iy), v in self.cells.items():
            vals[(iy - shift[1]) * nx + (ix - shift[0])] = v
        return {"origin": [round(ox, 3), round(oy, 3)], "cell_m": self.cell, "nx": nx, "ny": ny,
                "values": [round(v, 2) for v in vals], "max": round(max(vals), 2), "total_s": round(self.total, 2),
                "bounds_from": "room" if self.bounds else "dynamic"}


# ---------------------------------------------------------------- движок
class _TrackState:
    __slots__ = ("last_seen", "last_t", "last_meas_t", "last_xy", "zones", "lines")

    def __init__(self):
        self.last_seen = 0.0  # время последнего сообщения, где трек был
        self.last_t: float | None = None
        self.last_meas_t: float | None = None  # последнее время измеренной (не predicted) позиции
        self.last_xy: tuple[float, float] | None = None
        self.zones: dict[str, dict] = {}  # id зоны -> {inside, enter_t, enter_xy, cand_t, cand_xy}
        self.lines: dict[str, dict] = {}  # id линии -> {side, xy, s, t}


class Engine:
    def __init__(self, zones: list[dict], lines: list[dict], params: dict | None = None, room: dict | None = None, cell_m: float | None = None):
        self.p = {**DEFAULTS, **(params or {})}
        self.tracks: dict[tuple[str, int], _TrackState] = {}
        self.log: deque = deque(maxlen=int(self.p["events_max"]))
        self.n_events = 0
        self.heat = HeatMap(cell_m or self.p["cell_m"], room, self.p["max_extent_m"])
        self.zones: dict[str, dict] = {}
        self.lines: dict[str, dict] = {}
        self._sig: dict[str, tuple] = {}
        self._stats: dict[str, dict] = {}
        self.set_geometry(zones, lines)

    # --- геометрия ---
    def set_geometry(self, zones: list[dict], lines: list[dict]) -> None:
        """Обновить зоны/линии. У изменившихся по геометрии (и удалённых) сбрасывается состояние и счётчики."""
        new = {("z", z["id"]): ("z", tuple(map(tuple, z["points"]))) for z in zones}
        new.update({("l", l["id"]): ("l", tuple(l["a"]), tuple(l["b"])) for l in lines})
        gone = [k for k in self._sig if k not in new or self._sig[k] != new[k]]
        for kind, i in gone:
            for tr in self.tracks.values():
                (tr.zones if kind == "z" else tr.lines).pop(i, None)
            self._stats.pop(i, None)
        self._sig = new
        self.zones = {z["id"]: z for z in zones}
        self.lines = {l["id"]: l for l in lines}
        for i in self.zones:
            self._stats.setdefault(i, {"visits": 0, "exits": 0, "total_dwell_s": 0.0, "max_dwell_s": 0.0})
        for i in self.lines:
            self._stats.setdefault(i, {"a_to_b": 0, "b_to_a": 0})

    def reset_stats(self) -> None:
        """Обнулить счётчики и журнал (состояние «кто сейчас внутри» сохраняется)."""
        self.log.clear()
        for i, s in self._stats.items():
            for k in s:
                s[k] = 0 if isinstance(s[k], int) else 0.0

    # --- события ---
    def _emit(self, out: list, **ev) -> None:
        self.n_events += 1
        ev = {"n": self.n_events, **ev}
        out.append(ev)
        self.log.append(ev)

    def _enter(self, out, key, zid, t, xy):
        z = self.zones[zid]
        self._stats[zid]["visits"] += 1
        self._emit(out, type="enter", t=round(t, 4), kind=key[0], id=key[1], zone=zid, name=z["name"], pos=[round(xy[0], 3), round(xy[1], 3)])

    def _exit(self, out, key, zid, st, t, xy, reason=None):
        z = self.zones[zid]
        dwell = max(0.0, t - st["enter_t"])
        s = self._stats[zid]
        s["exits"] += 1
        s["total_dwell_s"] += dwell
        s["max_dwell_s"] = max(s["max_dwell_s"], dwell)
        self._emit(out, type="exit", t=round(t, 4), kind=key[0], id=key[1], zone=zid, name=z["name"],
                   dwell_s=round(dwell, 3), reason=reason, pos=[round(xy[0], 3), round(xy[1], 3)])

    def _close(self, out, key, tr: _TrackState, reason: str) -> None:
        t = tr.last_meas_t if tr.last_meas_t is not None else tr.last_seen
        for zid, st in list(tr.zones.items()):
            if st["inside"] and zid in self.zones:
                self._exit(out, key, zid, st, t, tr.last_xy or (0.0, 0.0), reason)
        del self.tracks[key]

    def close_all(self, reason: str = "end") -> list[dict]:
        out: list = []
        for key, tr in list(self.tracks.items()):
            self._close(out, key, tr, reason)
        return out

    # --- обработка кадра ---
    @staticmethod
    def _items(msg: dict):
        for p in msg.get("persons") or []:
            yield "person", p
        for o in msg.get("objects") or []:
            if isinstance(o, dict):
                yield (o.get("kind") or "object"), o

    def process(self, msg: dict) -> list[dict]:
        """Один кадр -> новые события. Побочно копит тепловую карту и счётчики."""
        t = msg.get("t")
        if not isinstance(t, (int, float)):
            return []
        t = float(t)
        h = self.p["hysteresis_m"] / 2
        tmin = self.p["min_time_s"]
        out: list = []
        seen = set()
        for kind, it in self._items(msg):
            tid = it.get("id")
            xy = it.get("floor") or it.get("pos")
            if not isinstance(tid, int) or not xy or len(xy) < 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in xy[:2]):
                continue
            x, y = float(xy[0]), float(xy[1])
            key = (kind, tid)
            seen.add(key)
            tr = self.tracks.get(key)
            if tr is None:
                tr = self.tracks[key] = _TrackState()
            tr.last_seen = t
            predicted = bool(it.get("predicted"))
            dt = min(max(0.0, t - tr.last_t), self.p["max_dt_s"]) if tr.last_t is not None else 0.0
            tr.last_t = t
            if predicted:
                continue  # прогноз: ни в карту, ни в события
            self.heat.add(x, y, dt)
            tr.last_meas_t, tr.last_xy = t, (x, y)
            for zid, z in self.zones.items():
                st = tr.zones.setdefault(zid, {"inside": False, "enter_t": 0.0, "cand_t": None, "cand_xy": None})
                d = poly_signed_dist(z["points"], x, y)
                want = (not st["inside"] and d >= h) or (st["inside"] and d <= -h)
                if not want:
                    st["cand_t"] = None
                    continue
                if st["cand_t"] is None:
                    st["cand_t"], st["cand_xy"] = t, (x, y)
                if t - st["cand_t"] >= tmin:
                    if st["inside"]:
                        self._exit(out, key, zid, st, st["cand_t"], st["cand_xy"])
                        st["inside"] = False
                    else:
                        st["inside"], st["enter_t"] = True, st["cand_t"]
                        self._enter(out, key, zid, st["cand_t"], st["cand_xy"])
                    st["cand_t"] = None
            for lid, ln in self.lines.items():
                ls = tr.lines.setdefault(lid, {"side": 0, "xy": None, "s": 0.0, "t": 0.0})
                s = line_side(ln["a"], ln["b"], x, y)
                if abs(s) < h:
                    continue
                side = 1 if s > 0 else -1
                if ls["side"] and side != ls["side"] and segment_param(ln["a"], ln["b"], ls["xy"], (x, y)) is not None:
                    frac = abs(ls["s"]) / (abs(ls["s"]) + abs(s))
                    ev_t = ls["t"] + frac * (t - ls["t"])
                    direction = "A→B" if side > 0 else "B→A"
                    self._stats[lid]["a_to_b" if side > 0 else "b_to_a"] += 1
                    self._emit(out, type="line_cross", t=round(ev_t, 4), kind=kind, id=tid, line=lid, name=ln["name"],
                               direction=direction, pos=[round(x, 3), round(y, 3)])
                ls["side"], ls["xy"], ls["s"], ls["t"] = side, (x, y), s, t
        for key, tr in list(self.tracks.items()):
            if key not in seen and t - tr.last_seen > self.p["lost_after_s"]:
                self._close(out, key, tr, "lost")
        return out

    # --- выдача ---
    def snapshot(self, t: float, events: list[dict] | None = None) -> dict:
        """Поле "zones" сообщения frame: кто сейчас внутри, счётчики линий, новые события."""
        inside: dict[str, list] = {zid: [] for zid in self.zones}
        for key, tr in self.tracks.items():
            for zid, st in tr.zones.items():
                if st["inside"] and zid in inside:
                    inside[zid].append({"kind": key[0], "id": key[1], "dwell_s": round(max(0.0, t - st["enter_t"]), 2)})
        for lst in inside.values():
            lst.sort(key=lambda w: (w["kind"], w["id"]))
        return {"inside": inside,
                "lines": {lid: {"a_to_b": self._stats[lid]["a_to_b"], "b_to_a": self._stats[lid]["b_to_a"]} for lid in self.lines},
                "events": events or []}

    def stats(self, t: float | None = None) -> dict:
        """Сводка: по зонам — сейчас внутри, посещений, суммарное/среднее/максимальное время (по завершённым выходам)."""
        snap = self.snapshot(t if t is not None else max((tr.last_seen for tr in self.tracks.values()), default=0.0))
        zs = []
        for zid, z in self.zones.items():
            s = self._stats[zid]
            zs.append({"id": zid, "name": z["name"], "color": z["color"], "inside": snap["inside"][zid], "inside_now": len(snap["inside"][zid]),
                       "visits": s["visits"], "exits": s["exits"], "total_dwell_s": round(s["total_dwell_s"], 2),
                       "mean_dwell_s": round(s["total_dwell_s"] / s["exits"], 2) if s["exits"] else None,
                       "max_dwell_s": round(s["max_dwell_s"], 2)})
        ls = [{"id": lid, "name": l["name"], "color": l["color"], **snap["lines"][lid],
               "total": snap["lines"][lid]["a_to_b"] + snap["lines"][lid]["b_to_a"]} for lid, l in self.lines.items()]
        return {"zones": zs, "lines": ls, "events_total": self.n_events}


# ---------------------------------------------------------------- офлайн и экспорт
CSV_COLUMNS = ["n", "t", "type", "kind", "id", "zone_or_line_id", "name", "direction", "dwell_s", "reason", "x", "y"]


def events_to_csv(events: list[dict]) -> str:
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    for e in events:
        w.writerow([e["n"], e["t"], e["type"], e["kind"], e["id"], e.get("zone") or e.get("line") or "", e.get("name", ""),
                    e.get("direction") or "", e.get("dwell_s", ""), e.get("reason") or "", *(e.get("pos") or ["", ""])[:2]])
    return buf.getvalue()


def analyze_session(path: str | Path, zones: list[dict], lines: list[dict], params: dict | None = None, cell_m: float | None = None) -> dict:
    """Прогнать запись сессии (.jsonl) через ту же логику: события, сводка по зонам/линиям, тепловая карта."""
    room = None
    msgs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue  # запись могла оборваться
            if not isinstance(m, dict):
                continue
            if m.get("type") == "hello" and room is None:
                room = m.get("room")
            elif m.get("type") == "frame":
                msgs.append(m)
    eng = Engine(zones, lines, params, room=room, cell_m=cell_m)
    events: list[dict] = []
    t0 = t1 = None
    for m in msgs:
        events += eng.process(m)
        if isinstance(m.get("t"), (int, float)):
            t0 = m["t"] if t0 is None else t0
            t1 = m["t"]
    events += eng.close_all("end")
    st = eng.stats(t1)
    return {"frames": len(msgs), "duration_s": round((t1 - t0), 2) if t0 is not None else 0.0,
            "events": events, "summary": {"zones": st["zones"], "lines": st["lines"]}, "heatmap": eng.heat.grid()}
