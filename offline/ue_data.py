"""Загрузка набора данных Unreal: камеры по кадрам, эталонная траектория, синхронное чтение видео.

Системы координат
-----------------
* Unreal (вход и выход): левосторонняя, X вперёд, Y вправо, Z вверх, сантиметры.
* Рабочая (внутри конвейера): правосторонняя, метры: (x, y, z) = (X, -Y, Z) / 100 —
  отражение оси Y делает систему правой, Z по-прежнему вверх. Преобразования: ue_to_world / world_to_ue.
* Камера OpenCV: x вправо, y вниз, z вперёд; X_cam = R @ X_world + t (как в backend/peleng/geometry.py).

Модель камеры Unreal (проверена проекцией эталона на все 5 камер):
строки матрицы M = FRotationMatrix(Pitch, Yaw, Roll) — оси камеры (вперёд, вправо, вверх) в мире Unreal;
d = M (P - C);  u = cx + f * d_y / d_x;  v = cy - f * d_z / d_x;  f = focal_mm / sensor_w_mm * W, без дисторсии.
В OpenCV-осях камеры: (x, y, z)_cv = (d_y, -d_z, d_x)  =>  R_cv<-ue = [M1; -M2; M0].
Для рабочей системы P_ue = 100 * S * P_w (S = diag(1,-1,1)), поэтому R = R_cv<-ue @ S, t = -R @ C_w.

Формат CSV: разделитель «;», кодировка cp1251 (или utf-8), CRLF, десятичный разделитель «,» или «.»;
справа в первых строках бывают служебные ячейки «Focal length;<мм>», «матрица;23,76*13,365мм».
Битые ячейки (артефакты Excel вида «авг.13») -> NaN -> линейная интерполяция по кадрам.
Строка k соответствует кадру видео k.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _paths  # noqa: F401
from peleng.geometry import CameraModel  # noqa: E402

DEFAULT_SENSOR_MM = (23.76, 13.365)
S_FLIP = np.diag([1.0, -1.0, 1.0])
CM_PER_M = 100.0


# ---------------------------------------------------------------------------
# Системы координат
# ---------------------------------------------------------------------------

def ue_to_world(P_ue: np.ndarray) -> np.ndarray:
    """Unreal (см, левая) -> рабочая (м, правая). Работает с (3,) и (N,3)."""
    return np.asarray(P_ue, dtype=np.float64) * np.array([1.0, -1.0, 1.0]) / CM_PER_M


def world_to_ue(P_w: np.ndarray) -> np.ndarray:
    """Рабочая (м, правая) -> Unreal (см, левая). Обратное к ue_to_world."""
    return np.asarray(P_w, dtype=np.float64) * np.array([1.0, -1.0, 1.0]) * CM_PER_M


def ue_rotmat(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """FRotationMatrix Unreal (градусы). Строки — оси камеры (вперёд X, вправо Y, вверх Z) в мире Unreal."""
    P, Y, R = np.radians([pitch, yaw, roll])
    sp, cp, sy, cy, sr, cr = math.sin(P), math.cos(P), math.sin(Y), math.cos(Y), math.sin(R), math.cos(R)
    X = [cp * cy, cp * sy, sp]
    Yv = [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp]
    Z = [-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp]
    return np.array([X, Yv, Z], dtype=np.float64)


def ue_camera_model(loc_ue, pitch, yaw, roll, focal_mm, size=(1920, 1080), sensor_w_mm=DEFAULT_SENSOR_MM[0]) -> CameraModel:
    """Камера Unreal -> CameraModel (OpenCV-оси, рабочая система мира)."""
    W, H = size
    f = focal_mm / sensor_w_mm * W
    K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1]], dtype=np.float64)
    M = ue_rotmat(pitch, yaw, roll)
    R = np.vstack([M[1], -M[2], M[0]]) @ S_FLIP
    C = ue_to_world(loc_ue)
    t = -R @ C
    return CameraModel(K=K, dist=np.zeros(5), size=(int(W), int(H)), R=R, t=t,
                       meta={"focal_mm": focal_mm, "ue_rot": (pitch, yaw, roll)})


# ---------------------------------------------------------------------------
# Парсер CSV
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?$")


def parse_num(s) -> float:
    """Число с «,» или «.»; всё прочее (пусто, «авг.13») -> NaN."""
    s = str(s).strip().replace(" ", "").replace(" ", "")
    if not _NUM_RE.match(s):
        return float("nan")
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return float("nan")


def _read_text(path: Path) -> str:
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1251", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


@dataclass
class Table:
    columns: dict[str, np.ndarray]  # имя -> значения по кадрам (индекс = номер кадра)
    meta: dict = field(default_factory=dict)
    bad_frames: list[int] = field(default_factory=list)

    def __getitem__(self, k: str) -> np.ndarray:
        return self.columns[k]

    @property
    def n(self) -> int:
        return len(self.columns["Frame"])


def read_ue_csv(path, wanted: tuple[str, ...], interpolate: bool = True) -> Table:
    """Устойчивое чтение CSV Unreal. Возвращает значения, индексированные номером кадра 0..max.

    interpolate=True — пропуски/битые ячейки заполняются линейной интерполяцией по кадрам."""
    text = _read_text(path).replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    sep = ";" if lines[0].count(";") >= lines[0].count(",") else ","
    hdr = [h.strip() for h in lines[0].split(sep)]
    meta: dict = {}
    # служебные пары «ключ;значение» в правых столбцах любых строк
    for ln in lines[:5]:
        cells = [c.strip() for c in ln.split(sep)]
        for i, c in enumerate(cells[:-1]):
            cl = c.lower()
            if cl.startswith("focal"):
                meta["focal_mm"] = parse_num(cells[i + 1])
            elif cl.startswith("матриц") or cl.startswith("sensor"):
                nums = re.findall(r"\d+(?:[.,]\d+)?", cells[i + 1])
                if len(nums) >= 2:
                    meta["sensor_mm"] = (parse_num(nums[0]), parse_num(nums[1]))
    col_idx = {}
    for name in ("Frame",) + tuple(wanted):
        if name in hdr:
            col_idx[name] = hdr.index(name)
        else:
            raise ValueError(f"{path}: нет столбца {name}")
    rows = []
    for ln in lines[1:]:
        cells = ln.split(sep)
        rows.append([parse_num(cells[j]) if j < len(cells) else float("nan") for j in col_idx.values()])
    arr = np.array(rows, dtype=np.float64)
    names = list(col_idx)
    frames = arr[:, 0]
    # номер кадра: если битый — восстановить по порядку строк
    for i in range(len(frames)):
        if not np.isfinite(frames[i]):
            frames[i] = frames[i - 1] + 1 if i > 0 else 0
    frames = frames.astype(int)
    nmax = int(frames.max()) + 1
    out = {n: np.full(nmax, np.nan) for n in names[1:]}
    seen = np.zeros(nmax, bool)
    for r, fr in zip(arr, frames):
        if fr < 0 or seen[fr]:
            continue  # дубликаты строк (например, последняя строка) — берём первую
        seen[fr] = True
        for j, n in enumerate(names[1:], start=1):
            out[n][fr] = r[j]
    bad = sorted({int(i) for n in out for i in np.where(~np.isfinite(out[n]))[0]})
    idx = np.arange(nmax)
    for n, v in out.items():
        ok = np.isfinite(v)
        if ok.sum() == 0:
            continue
        if interpolate and not ok.all():
            out[n] = np.interp(idx, idx[ok], v[ok])
    out["Frame"] = idx
    return Table(out, meta, bad)


def write_ue_csv(path, frames: np.ndarray, P_ue: np.ndarray) -> None:
    """Траектория в формате эталона: Frame;Loc_X;Loc_Y;Loc_Z (см Unreal, точка — десятичный разделитель)."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("Frame;Loc_X;Loc_Y;Loc_Z\r\n")
        for k, p in zip(frames, P_ue):
            if not np.all(np.isfinite(p)):
                f.write(f"{int(k)};;;\r\n")
            else:
                f.write(f"{int(k)};{p[0]:.4f};{p[1]:.4f};{p[2]:.4f}\r\n")


# ---------------------------------------------------------------------------
# Набор данных
# ---------------------------------------------------------------------------

_CAM_RE = re.compile(r"object_transform_(.+?)(\d+)\.csv$", re.I)


@dataclass
class UECamera:
    cam_id: int
    name: str
    csv_path: Path
    video_path: Path | None
    table: Table
    focal_mm: float
    sensor_mm: tuple[float, float]
    size: tuple[int, int] = (1920, 1080)
    _cache: dict = field(default_factory=dict, repr=False)

    def model(self, k: int) -> CameraModel:
        """CameraModel для кадра k (поза может меняться от кадра к кадру)."""
        k = int(min(max(k, 0), self.table.n - 1))
        key = self._pose_key(k)
        m = self._cache.get(key)
        if m is None:
            t = self.table
            m = ue_camera_model((t["Loc_X"][k], t["Loc_Y"][k], t["Loc_Z"][k]), t["Rot_Pitch"][k], t["Rot_Yaw"][k],
                                t["Rot_Roll"][k], self.focal_mm, self.size, self.sensor_mm[0])
            self._cache[key] = m
        return m

    def _pose_key(self, k: int) -> tuple:
        t = self.table
        return tuple(round(float(t[c][k]), 5) for c in ("Loc_X", "Loc_Y", "Loc_Z", "Rot_Pitch", "Rot_Yaw", "Rot_Roll"))

    def is_static(self, k0: int, k1: int) -> bool:
        """Поза не менялась на отрезке кадров [k0, k1]."""
        k0, k1 = max(k0, 0), min(k1, self.table.n - 1)
        return all(self._pose_key(k) == self._pose_key(k0) for k in range(k0, k1 + 1))


@dataclass
class UEDataset:
    root: Path
    cameras: list[UECamera]
    truth_ue: np.ndarray | None  # (N,3) эталон в см Unreal или None
    truth_bad_frames: list[int]
    n_frames: int
    fps: float = 30.0

    def truth_world(self) -> np.ndarray | None:
        return None if self.truth_ue is None else ue_to_world(self.truth_ue)


def _find_video(root: Path, name: str, num: str) -> Path | None:
    cands = sorted(root.glob(f"*{name}{num}.mp4")) + sorted(root.glob(f"*{name}{num}.*"))
    cands = [c for c in cands if c.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")]
    return cands[0] if cands else None


def load_dataset(root, truth_name: str | None = None, log=print) -> UEDataset:
    """Найти в папке CSV камер (object_transform_<Имя><N>.csv), видео (*<Имя><N>.mp4) и эталон (CSV без Rot_*)."""
    import cv2

    root = Path(root)
    cams: list[UECamera] = []
    truth = None
    truth_bad: list[int] = []
    for p in sorted(root.glob("*.csv")):
        head = _read_text(p).splitlines()[0] if p.stat().st_size else ""
        is_cam = "Rot_Yaw" in head
        if is_cam:
            m = _CAM_RE.search(p.name)
            if not m:
                continue
            name, num = m.group(1), m.group(2)
            tab = read_ue_csv(p, ("Loc_X", "Loc_Y", "Loc_Z", "Rot_Roll", "Rot_Pitch", "Rot_Yaw"))
            vid = _find_video(root, name, num)
            size = (1920, 1080)
            if vid is not None:
                cap = cv2.VideoCapture(str(vid))
                w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                if w > 0 and h > 0:
                    size = (w, h)
            cams.append(UECamera(int(num), name + num, p, vid, tab, float(tab.meta.get("focal_mm", 35.0)),
                                 tuple(tab.meta.get("sensor_mm", DEFAULT_SENSOR_MM)), size))
        elif "Loc_X" in head and (truth_name is None or p.name == truth_name):
            tab = read_ue_csv(p, ("Loc_X", "Loc_Y", "Loc_Z"))
            truth = np.stack([tab["Loc_X"], tab["Loc_Y"], tab["Loc_Z"]], axis=1)
            truth_bad = tab.bad_frames
    cams.sort(key=lambda c: c.cam_id)
    if not cams:
        raise FileNotFoundError(f"в {root} нет CSV камер object_transform_*N.csv")
    # число кадров: по самому длинному видео (камеры с более коротким видео/CSV на хвосте просто «не видят»;
    # поза камеры за концом её CSV — последняя известная), но не больше самого длинного CSV
    n_csv = max(c.table.n for c in cams)
    nv = []
    fps = 30.0
    for c in cams:
        if c.video_path is None:
            log(f"внимание: для камеры {c.name} нет видео — она не участвует в детекции")
            continue
        cap = cv2.VideoCapture(str(c.video_path))
        cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or cnt <= 0:
            log(f"внимание: видео {c.video_path.name} не открывается — камера {c.name} без детекций")
            c.video_path = None
        else:
            nv.append(cnt)
            fps = cap.get(cv2.CAP_PROP_FPS) or fps
            if cnt != c.table.n:
                log(f"внимание: камера {c.name}: кадров в видео {cnt}, строк в CSV {c.table.n}")
        cap.release()
        if not np.isfinite(c.focal_mm) or c.focal_mm <= 0:
            log(f"внимание: камера {c.name}: фокус не найден в CSV, принят 35 мм")
            c.focal_mm = 35.0
        if c.table.bad_frames:
            log(f"камера {c.name}: битые ячейки в кадрах {c.table.bad_frames[:10]} — заполнены интерполяцией")
    n = min(n_csv, max(nv)) if nv else n_csv
    if truth is not None and truth_bad:
        log(f"эталон: битые ячейки в кадрах {truth_bad[:10]} — заполнены интерполяцией")
    return UEDataset(root, cams, truth, truth_bad, n, fps)


class SyncVideoReader:
    """Синхронное покадровое чтение нескольких видео (строго вперёд — причинно)."""

    def __init__(self, paths: list[Path], n_frames: int | None = None):
        import cv2

        self.caps = [cv2.VideoCapture(str(p)) if p is not None else None for p in paths]
        self.n = n_frames

    def __iter__(self):
        k = 0
        while self.n is None or k < self.n:
            imgs = []
            for cap in self.caps:
                ok, img = cap.read() if cap is not None else (False, None)
                imgs.append(img if ok else None)
            if all(i is None for i in imgs) and self.n is None:
                break  # при заданном числе кадров отдаём пустые кадры до конца (видео короче CSV)
            yield k, imgs
            k += 1
        for cap in self.caps:
            if cap is not None:
                cap.release()
