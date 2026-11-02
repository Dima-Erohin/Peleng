"""Восстановление траектории после записи: сшивка обрывков, сглаживание вперёд-назад, заполнение пропусков, экспорт.

Живой фильтр Калмана (tracking.py) видит только прошлое. После записи доступна вся сессия, поэтому:
  1. треки из записи (.jsonl) собираются по ключу (kind, id) — "persons" (kind="person") и "objects";
  2. обрывки одного человека/объекта сшиваются: трек кончился, другой того же kind начался через ≤ max_gap_s
     в месте, согласном с его скоростью;
  3. фильтр Калмана (постоянная скорость, независимо по осям) + сглаживатель Рауха–Тунга–Штрибеля (RTS):
     каждая точка оценивается по прошлому и будущему; веса по mode (tri точнее mono), predicted-точки — не измерения;
  4. результат — на равномерной сетке (rate_hz); пропуски ≤ max_gap_s заполняются тем же сглаживателем
     (source="interpolated"), у каждой точки σ из ковариации сглаживателя;
  5. метрики трека и экспорт CSV/JSON (сырые и восстановленные данные).
"""
from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULTS: dict = {
    "max_gap_s": 1.5,  # дольше — трек рвётся (не сшиваем и не заполняем); как max_age_s живого трекера
    "rate_hz": 30.0,  # частота равномерной сетки на выходе
    "accel_std": 1.5,  # м/с² — шум ускорения модели постоянной скорости (меньше — глаже)
    "meas_std": {"tri": [0.05, 0.05], "mono": [0.20, 0.30]},  # σ измерения [xy, z], м — как в tracking.py
    "reproj_ref_px": 8.0,  # tri с ошибкой репроекции больше этой — вес меньше (σ растёт пропорционально, до ×4)
    "stitch_gate_m": 0.5,  # допуск на расхождение места при сшивке, м …
    "stitch_slack_mps": 1.0,  # … плюс столько метров на каждую секунду пропуска
    "min_points": 3,  # треки с меньшим числом измерений не восстанавливаем (мусор)
}

CSV_COLUMNS = ["t", "kind", "id", "x", "y", "z", "vx", "vy", "vz", "speed", "source", "std"]


def settings(cfg: dict | None = None, **override) -> dict:
    """Настройки: DEFAULTS ← cfg["trajectory"] ← явные параметры (None игнорируются)."""
    s = {**DEFAULTS, **((cfg or {}).get("trajectory", {}) or {})}
    s.update({k: v for k, v in override.items() if v is not None})
    s["max_gap_s"] = float(s["max_gap_s"])
    s["rate_hz"] = float(s["rate_hz"])
    if not (0 < s["rate_hz"] <= 1000) or s["max_gap_s"] < 0:
        raise ValueError("rate_hz должен быть в (0, 1000], max_gap_s ≥ 0")
    return s


# ---------------------------------------------------------------- загрузка записи
@dataclass
class RawTrack:
    kind: str  # "person" | "object" | …
    id: int
    t: np.ndarray  # (N,) время, с
    pos: np.ndarray  # (N, 3)
    mode: list[str]  # tri | mono
    predicted: np.ndarray  # (N,) bool — точка из прогноза живого фильтра, не измерение
    reproj_err_px: np.ndarray  # (N,) nan — нет данных
    vel: np.ndarray  # (N, 3) скорость из записи (nan — нет)
    pos_std: np.ndarray  # (N, 3) pos_std_m из записи (nan — нет)
    cls: str | None = None  # класс объекта (для objects)
    merged_ids: list[int] = field(default_factory=list)  # какие ID записи вошли в трек после сшивки

    def __post_init__(self):
        if not self.merged_ids:
            self.merged_ids = [self.id]

    @property
    def measured(self) -> np.ndarray:
        return ~self.predicted

    def __len__(self) -> int:
        return len(self.t)


def _vec3(v) -> list[float]:
    if isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
        return [float(x) for x in v]
    return [math.nan] * 3


def load_session(path: str | Path) -> tuple[dict | None, list[RawTrack]]:
    """Запись сессии -> (hello или None, треки). Битые строки пропускаются (запись могла оборваться)."""
    hello = None
    rows: dict[tuple[str, int], list] = {}
    cls_of: dict[tuple[str, int], str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(m, dict):
                continue
            if m.get("type") == "hello" and hello is None:
                hello = m
                continue
            if m.get("type") != "frame" or not isinstance(m.get("t"), (int, float)):
                continue
            t = float(m["t"])
            items = [("person", p) for p in m.get("persons") or []]
            items += [(o.get("kind") or "object", o) for o in m.get("objects") or [] if isinstance(o, dict)]
            for kind, it in items:
                pos = _vec3(it.get("pos"))
                if not isinstance(it.get("id"), int) or math.isnan(pos[0]):
                    continue
                key = (kind, it["id"])
                rp = it.get("reproj_err_px")
                rows.setdefault(key, []).append(
                    (t, pos, it.get("mode") or "tri", bool(it.get("predicted", False)),
                     float(rp) if isinstance(rp, (int, float)) else math.nan, _vec3(it.get("vel")), _vec3(it.get("pos_std_m")))
                )
                if it.get("cls"):
                    cls_of[key] = str(it["cls"])
    tracks = []
    for (kind, tid), r in rows.items():
        r.sort(key=lambda x: x[0])
        # одно время — одна точка (на случай повторов)
        uniq = [r[0]] + [b for a, b in zip(r, r[1:]) if b[0] > a[0]]
        tracks.append(_track_from_rows(kind, tid, uniq, cls_of.get((kind, tid))))
    tracks.sort(key=lambda tr: (tr.t[0], tr.kind, tr.id))
    return hello, tracks


def _track_from_rows(kind: str, tid: int, r: list, cls: str | None) -> RawTrack:
    return RawTrack(
        kind=kind, id=tid,
        t=np.array([x[0] for x in r]),
        pos=np.array([x[1] for x in r], float).reshape(-1, 3),
        mode=[x[2] for x in r],
        predicted=np.array([x[3] for x in r], bool),
        reproj_err_px=np.array([x[4] for x in r], float),
        vel=np.array([x[5] for x in r], float).reshape(-1, 3),
        pos_std=np.array([x[6] for x in r], float).reshape(-1, 3),
        cls=cls,
    )


# ---------------------------------------------------------------- сшивка обрывков
def _end_state(tr: RawTrack, at_end: bool, window_s: float = 0.5) -> tuple[float, np.ndarray, np.ndarray]:
    """(время, положение, скорость) на конце (или в начале) трека по измерениям — прямая по последним ~0.5 с."""
    m = tr.measured
    t, p = tr.t[m], tr.pos[m]
    sel = t >= t[-1] - window_s if at_end else t <= t[0] + window_s
    ts, ps = t[sel], p[sel]
    t_ref = t[-1] if at_end else t[0]
    if len(ts) >= 3 and ts[-1] - ts[0] > 0.1:
        A = np.column_stack([ts - t_ref, np.ones(len(ts))])
        coef, *_ = np.linalg.lstsq(A, ps, rcond=None)
        return t_ref, coef[1], coef[0]
    return t_ref, (p[-1] if at_end else p[0]), np.zeros(3)


def stitch(tracks: list[RawTrack], max_gap_s: float, gate_m: float = 0.5, slack_mps: float = 1.0) -> list[RawTrack]:
    """Сшить обрывки: конец A -> начало B того же kind (и класса), пропуск 0 ≤ gap ≤ max_gap_s,
    начало B рядом с продолжением A по его скорости (по горизонтали). Жадно, начиная с лучших пар;
    у каждого трека не больше одного продолжения."""
    good = [tr for tr in tracks if tr.measured.any()]
    ends = {i: _end_state(tr, True) for i, tr in enumerate(good)}
    starts = {i: _end_state(tr, False) for i, tr in enumerate(good)}
    cand = []
    for i, a in enumerate(good):
        ta, pa, va = ends[i]
        for j, b in enumerate(good):
            if i == j or a.kind != b.kind or (a.cls and b.cls and a.cls != b.cls):
                continue
            tb, pb, _ = starts[j]
            gap = tb - ta
            if gap < -1e-6 or gap > max_gap_s:
                continue
            gap = max(gap, 0.0)
            d = float(np.linalg.norm((pb - (pa + va * gap))[:2]))
            gate = gate_m + slack_mps * gap
            if d <= gate:
                cand.append((d / gate, i, j))
    nxt: dict[int, int] = {}
    prv: dict[int, int] = {}
    for _, i, j in sorted(cand):
        if i in nxt or j in prv:
            continue
        # не замыкать цепочку в кольцо
        k = i
        while k in prv:
            k = prv[k]
        if k == j:
            continue
        nxt[i], prv[j] = j, i
    out = []
    for i, tr in enumerate(good):
        if i in prv:
            continue
        chain = [tr]
        k = i
        while k in nxt:
            k = nxt[k]
            chain.append(good[k])
        out.append(_concat(chain))
    out += [tr for tr in tracks if not tr.measured.any()]
    out.sort(key=lambda tr: (tr.t[0], tr.kind, tr.id))
    return out


def _concat(chain: list[RawTrack]) -> RawTrack:
    if len(chain) == 1:
        return chain[0]
    parts = []
    for k, tr in enumerate(chain):
        keep = np.ones(len(tr), bool)
        if k < len(chain) - 1:  # хвост прогноза до начала следующего обрывка — выкинуть
            keep &= tr.t <= chain[k + 1].t[0] - 1e-9
        parts.append((tr, keep))
    first = chain[0]
    t = np.concatenate([tr.t[k] for tr, k in parts])
    order = np.argsort(t, kind="stable")
    res = RawTrack(
        kind=first.kind, id=first.id,
        t=t[order],
        pos=np.concatenate([tr.pos[k] for tr, k in parts])[order],
        mode=[m for tr, k in parts for m, kk in zip(tr.mode, k) if kk],
        predicted=np.concatenate([tr.predicted[k] for tr, k in parts])[order],
        reproj_err_px=np.concatenate([tr.reproj_err_px[k] for tr, k in parts])[order],
        vel=np.concatenate([tr.vel[k] for tr, k in parts])[order],
        pos_std=np.concatenate([tr.pos_std[k] for tr, k in parts])[order],
        cls=next((tr.cls for tr in chain if tr.cls), None),
        merged_ids=[i for tr in chain for i in tr.merged_ids],
    )
    res.mode = [res.mode[i] for i in order]
    return res


# ---------------------------------------------------------------- сглаживание (Калман + RTS)
def _meas_std(tr: RawTrack, s: dict) -> np.ndarray:
    """σ каждого измерения по осям (N, 3): по mode, с поправкой на ошибку репроекции и pos_std_m (если есть)."""
    ms = s["meas_std"]
    out = np.zeros((len(tr), 3))
    for i, mode in enumerate(tr.mode):
        xy, z = ms.get(mode, ms["mono"])
        k = 1.0
        rp = tr.reproj_err_px[i]
        if mode == "tri" and not math.isnan(rp):
            k = min(4.0, max(1.0, rp / s["reproj_ref_px"]))
        out[i] = [xy * k, xy * k, z * k]
    ps = tr.pos_std
    has = ~np.isnan(ps)
    out[has] = np.maximum(out[has], ps[has])
    return out


def smooth_rts(tm: np.ndarray, zm: np.ndarray, sm: np.ndarray, tq: np.ndarray, accel_std: float):
    """Калман (постоянная скорость, оси независимы) по измерениям (tm, zm, σ sm) + RTS-сглаживание.
    Возвращает оценку в моменты tq: (pos (Q,3), vel (Q,3), σ pos (Q,3), σ vel (Q,3))."""
    M, Q = len(tm), len(tq)
    T = np.concatenate([tm, tq])
    is_q = np.concatenate([np.zeros(M, bool), np.ones(Q, bool)])
    src = np.concatenate([np.arange(M), np.arange(Q)])
    order = np.lexsort((is_q, T))  # по времени; при равенстве — сначала измерение
    T, is_q, src = T[order], is_q[order], src[order]
    n = len(T)
    q = accel_std**2
    xp = np.zeros((n, 3, 2)); Pp = np.zeros((n, 3, 2, 2))
    xf = np.zeros((n, 3, 2)); Pf = np.zeros((n, 3, 2, 2))
    x = np.zeros((3, 2)); x[:, 0] = zm[0]
    P = np.zeros((3, 2, 2)); P[:, 0, 0] = sm[0] ** 2; P[:, 1, 1] = 1.0**2
    t_prev = T[0]
    for k in range(n):
        dt = T[k] - t_prev
        t_prev = T[k]
        if dt > 0:
            F = np.array([[1.0, dt], [0.0, 1.0]])
            Qm = q * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
            x = x @ F.T
            P = F @ P @ F.T + Qm
        xp[k], Pp[k] = x, P
        if not is_q[k]:
            j = src[k]
            S = P[:, 0, 0] + sm[j] ** 2
            K = P[:, :, 0] / S[:, None]  # (3, 2)
            y = zm[j] - x[:, 0]
            x = x + K * y[:, None]
            P = P - K[:, :, None] * P[:, 0, :][:, None, :]
        xf[k], Pf[k] = x, P
    xs, Ps = xf.copy(), Pf.copy()
    for k in range(n - 2, -1, -1):
        dt = T[k + 1] - T[k]
        F = np.array([[1.0, dt], [0.0, 1.0]])
        C = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1] + np.eye(2) * 1e-15)  # (3,2,2)
        xs[k] = xf[k] + np.einsum("aij,aj->ai", C, xs[k + 1] - xp[k + 1])
        Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ np.transpose(C, (0, 2, 1))
    sel = np.flatnonzero(is_q)
    out_idx = np.empty(Q, int)
    out_idx[src[sel]] = sel
    xs, Ps = xs[out_idx], Ps[out_idx]
    return xs[:, :, 0], xs[:, :, 1], np.sqrt(np.maximum(Ps[:, :, 0, 0], 0)), np.sqrt(np.maximum(Ps[:, :, 1, 1], 0))


@dataclass
class Restored:
    kind: str
    id: int
    merged_ids: list[int]
    cls: str | None
    t: np.ndarray  # (N,) равномерная сетка (с разрывами на пропусках > max_gap_s)
    pos: np.ndarray  # (N, 3)
    vel: np.ndarray  # (N, 3)
    source: list[str]  # measured | interpolated
    std: np.ndarray  # (N,) σ положения (среднеквадратичная по осям), м
    metrics: dict


def _speed(kind: str, v: np.ndarray) -> np.ndarray:
    """Скорость: у людей горизонтальная (как "speed" в frame), у объектов — полная 3D."""
    v = np.asarray(v, float)
    return np.linalg.norm(v[..., :2] if kind == "person" else v, axis=-1)


def restore_track(tr: RawTrack, s: dict) -> Restored | None:
    m = tr.measured
    if m.sum() < s["min_points"]:
        return None
    tm, zm, sm = tr.t[m], tr.pos[m], _meas_std(tr, s)[m]
    rate, max_gap = s["rate_hz"], s["max_gap_s"]
    dts = np.diff(tm)
    # «пропуск» — интервал между измерениями заметно длиннее обычного
    gap_thr = max(2.5 * float(np.median(dts)) if len(dts) else 0.0, 2.0 / rate)
    breaks = np.flatnonzero(dts > max_gap)  # здесь трек рвётся: не заполняем
    seg_bounds = np.split(np.arange(len(tm)), breaks + 1)
    T, Pp, V, S, SRC = [], [], [], [], []
    gaps = []
    for idx in seg_bounds:
        t0, t1 = tm[idx[0]], tm[idx[-1]]
        tg = np.arange(math.ceil(t0 * rate - 1e-9), math.floor(t1 * rate + 1e-9) + 1) / rate
        if len(tg) == 0:
            tg = tm[idx]
        p, v, sp, _ = smooth_rts(tm[idx], zm[idx], sm[idx], tg, s["accel_std"])
        ts = tm[idx]
        k = np.clip(np.searchsorted(ts, tg), 1, max(1, len(ts) - 1))
        if len(ts) >= 2:
            inside_gap = (ts[k] - ts[k - 1] > gap_thr) & (tg > ts[k - 1] + 1e-9) & (tg < ts[k] - 1e-9)
        else:
            inside_gap = np.zeros(len(tg), bool)
        d = np.diff(ts)
        gaps += [float(g) for g in d[d > gap_thr]]
        T.append(tg); Pp.append(p); V.append(v); S.append(np.sqrt((sp**2).mean(axis=1)))
        SRC += ["interpolated" if g else "measured" for g in inside_gap]
    t = np.concatenate(T)
    pos, vel, std = np.concatenate(Pp), np.concatenate(V), np.concatenate(S)
    metrics = track_metrics(tr.kind, t, pos, vel, T)
    raw_pts = zm if tr.kind != "person" else zm[:, :2]
    metrics.update({
        "points_raw": int(len(tr)),
        "points_measured": int(m.sum()),
        "points_predicted": int((~m).sum()),
        "path_raw_m": round(float(np.linalg.norm(np.diff(raw_pts, axis=0), axis=1).sum()), 3),
        "gaps_filled": len(gaps),
        "gaps_filled_s": round(float(sum(gaps)), 3),
        "gap_max_s": round(float(max(gaps)), 3) if gaps else 0.0,
        "breaks": int(len(breaks)),  # пропуски длиннее max_gap_s — не заполнены
        "points_interpolated": int(sum(x == "interpolated" for x in SRC)),
        "std_mean_m": round(float(std.mean()), 4),
        "mode_tri_share": round(sum(1 for i in np.flatnonzero(m) if tr.mode[i] == "tri") / max(1, int(m.sum())), 3),
    })
    return Restored(tr.kind, tr.id, list(tr.merged_ids), tr.cls, t, pos, vel, SRC, std, metrics)


def track_metrics(kind: str, t: np.ndarray, pos: np.ndarray, vel: np.ndarray, segments: list[np.ndarray]) -> dict:
    """Длительность (по участкам без разрывов), путь по сглаженной траектории, средняя/макс. скорость."""
    dur = float(sum(seg[-1] - seg[0] for seg in segments if len(seg)))
    path = 0.0
    i = 0
    for seg in segments:
        p = pos[i:i + len(seg)]
        p = p if kind != "person" else p[:, :2]
        path += float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        i += len(seg)
    sp = _speed(kind, vel)
    return {
        "t_start": round(float(t[0]), 4),
        "t_end": round(float(t[-1]), 4),
        "duration_s": round(dur, 3),
        "path_m": round(path, 3),
        "speed_mean_mps": round(path / dur, 3) if dur > 0 else 0.0,
        "speed_max_mps": round(float(sp.max()), 3) if len(sp) else 0.0,
    }


# ---------------------------------------------------------------- вся сессия
@dataclass
class SessionResult:
    hello: dict | None
    raw: list[RawTrack]  # как в записи (без сшивки)
    restored: list[Restored]
    params: dict


def restore_tracks(raw: list[RawTrack], s: dict) -> list[Restored]:
    stitched = stitch(raw, s["max_gap_s"], s["stitch_gate_m"], s["stitch_slack_mps"])
    out = [r for r in (restore_track(tr, s) for tr in stitched) if r is not None]
    out.sort(key=lambda r: (r.kind, r.id))
    return out


def restore_session(path: str | Path, cfg: dict | None = None, **override) -> SessionResult:
    s = settings(cfg, **override)
    hello, raw = load_session(path)
    params = {k: s[k] for k in ("max_gap_s", "rate_hz", "accel_std", "stitch_gate_m", "stitch_slack_mps", "min_points")}
    return SessionResult(hello, raw, restore_tracks(raw, s), params)


def summary(res: SessionResult) -> dict:
    """Короткая сводка для интерфейса/консоли (без рядов точек)."""
    t_all = [tr.t for tr in res.raw]
    return {
        "params": res.params,
        "raw_tracks": len(res.raw),
        "tracks": len(res.restored),
        "t0": round(float(min(x[0] for x in t_all)), 4) if t_all else None,
        "t1": round(float(max(x[-1] for x in t_all)), 4) if t_all else None,
        "items": [{"kind": r.kind, "id": r.id, "merged_ids": r.merged_ids, "cls": r.cls, **r.metrics} for r in res.restored],
    }


# ---------------------------------------------------------------- экспорт
def _f(v, nd: int = 4) -> str:
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{float(v):.{nd}f}"


def rows_restored(res: SessionResult):
    for r in res.restored:
        sp = _speed(r.kind, r.vel)
        for i in range(len(r.t)):
            yield [_f(r.t[i]), r.kind, r.id, *(_f(x) for x in r.pos[i]), *(_f(x) for x in r.vel[i]), _f(sp[i]), r.source[i], _f(r.std[i])]


def rows_raw(res: SessionResult):
    rows = []
    for tr in res.raw:
        sp = _speed(tr.kind, tr.vel)
        std = np.sqrt((tr.pos_std**2).mean(axis=1))  # nan, если pos_std_m в записи нет
        for i in range(len(tr)):
            rows.append([tr.t[i], tr.kind, tr.id, *(_f(x) for x in tr.pos[i]), *(_f(x) for x in tr.vel[i]), _f(sp[i]),
                         "predicted" if tr.predicted[i] else tr.mode[i], _f(std[i])])
    rows.sort(key=lambda r: (r[0], r[2]))
    for r in rows:
        r[0] = _f(r[0])
        yield r


def to_csv(res: SessionResult, raw: bool = False) -> str:
    """CSV: t, kind, id, x, y, z, vx, vy, vz, speed, source, std.
    Восстановленные: source = measured | interpolated. Сырые: source = tri | mono | predicted, std — из pos_std_m (если есть)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    w.writerows(rows_raw(res) if raw else rows_restored(res))
    return buf.getvalue()


def _rl(a: np.ndarray, nd: int = 4) -> list:
    return [None if isinstance(x, float) and math.isnan(x) else x for x in np.round(np.asarray(a, float), nd).tolist()] if np.ndim(a) == 1 \
        else [[None if math.isnan(x) else x for x in row] for row in np.round(np.asarray(a, float), nd).tolist()]


def to_json(res: SessionResult, raw: bool = False, step: int = 1) -> dict:
    """JSON-экспорт. step > 1 — прореживание точек (для отображения во вьювере)."""
    if raw:
        tracks = [{
            "kind": tr.kind, "id": tr.id, "cls": tr.cls,
            "t": _rl(tr.t[::step]), "pos": _rl(tr.pos[::step]), "vel": _rl(tr.vel[::step]),
            "mode": tr.mode[::step], "predicted": tr.predicted[::step].tolist(),
            "reproj_err_px": _rl(tr.reproj_err_px[::step], 2),
        } for tr in res.raw]
    else:
        tracks = [{
            "kind": r.kind, "id": r.id, "merged_ids": r.merged_ids, "cls": r.cls, "metrics": r.metrics,
            "t": _rl(r.t[::step]), "pos": _rl(r.pos[::step]), "vel": _rl(r.vel[::step]),
            "source": r.source[::step], "std": _rl(r.std[::step]),
        } for r in res.restored]
    return {"format": "peleng-trajectory/1", "data": "raw" if raw else "restored", "units": "m, s", "up": "z",
            "params": res.params, "tracks": tracks}
