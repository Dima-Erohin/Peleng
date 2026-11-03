"""Экспорт результатов прогона в один JSON для фронтенда (React / Three.js).

    python offline/export_viewer.py --data <папка данных> --run <папка результатов run.py> [--out viewer.json]

Системы координат в файле:
  * "ue"    — Unreal: см, левая система, X вперёд, Y вправо, Z вверх (как во входных CSV и в trajectory.csv);
  * "three" — Three.js: метры, правая система, Y вверх: (x, y, z)_three = (X_ue, Z_ue, Y_ue) / 100.
    (Это рабочая правая система конвейера (X, -Y, Z)/100, повёрнутая так, чтобы «вверх» был +Y.)
Камера в Three.js: position + quaternion [x, y, z, w] для THREE.PerspectiveCamera (смотрит вдоль своей -Z,
вверх — её +Y), fov_v_deg — вертикальный угол обзора, aspect = width / height.
Формат описан в offline/README.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "offline"

from .ue_data import load_dataset, read_ue_csv  # noqa: E402

# рабочая (x, y, z-вверх) -> three (x, z, -y)
W2THREE = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])


def ue_to_three(P_ue) -> np.ndarray:
    P = np.asarray(P_ue, dtype=np.float64)
    return np.stack([P[..., 0], P[..., 2], P[..., 1]], -1) / 100.0


def _quat(Rm: np.ndarray) -> list[float]:
    """Матрица поворота -> кватернион [x, y, z, w]."""
    m = Rm
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return [round(float(v), 7) for v in (x, y, z, w)]


def three_camera(cm) -> dict:
    """CameraModel (OpenCV-оси, рабочая система) -> позиция и кватернион THREE.PerspectiveCamera."""
    R = cm.R  # строки: оси камеры x (вправо), y (вниз), z (вперёд) в рабочей системе
    axes = np.stack([R[0], -R[1], -R[2]], 1)  # столбцы: X_three-камеры = вправо, Y = вверх, Z = назад
    Rt = W2THREE @ axes
    pos = W2THREE @ cm.center
    return {"position": [round(float(v), 4) for v in pos], "quaternion": _quat(Rt)}


def _r(v, nd=2):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def build(data_dir, run_dir) -> dict:
    ds = load_dataset(data_dir)
    run_dir = Path(run_dir)
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8")) if (run_dir / "run_info.json").exists() else {}
    diag = list(csv.DictReader(open(run_dir / "trajectory_diag.csv", encoding="utf-8"), delimiter=";"))
    n = len(diag)
    cams = []
    for c in ds.cameras:
        m0 = c.model(0)
        t = c.table
        static = c.is_static(0, n - 1)
        cam = {
            "id": c.cam_id, "name": c.name, "video": c.video_path.name if c.video_path else None,
            "width": c.size[0], "height": c.size[1], "focal_mm": c.focal_mm, "sensor_mm": list(c.sensor_mm),
            "fx_px": round(float(m0.K[0, 0]), 3), "fy_px": round(float(m0.K[1, 1]), 3),
            "cx_px": float(m0.K[0, 2]), "cy_px": float(m0.K[1, 2]),
            "fov_v_deg": round(math.degrees(2 * math.atan(c.size[1] / 2 / m0.K[1, 1])), 4),
            "fov_h_deg": round(math.degrees(2 * math.atan(c.size[0] / 2 / m0.K[0, 0])), 4),
            "static": static,
            "ue": {"loc_cm": [float(t["Loc_X"][0]), float(t["Loc_Y"][0]), float(t["Loc_Z"][0])],
                   "rot_deg": {"pitch": float(t["Rot_Pitch"][0]), "yaw": float(t["Rot_Yaw"][0]), "roll": float(t["Rot_Roll"][0])}},
            "three": three_camera(m0),
        }
        if not static:
            cam["per_frame"] = {
                "ue_yaw_deg": [round(float(v), 4) for v in t["Rot_Yaw"][:n]],
                "ue_pitch_deg": [round(float(v), 4) for v in t["Rot_Pitch"][:n]],
                "ue_roll_deg": [round(float(v), 4) for v in t["Rot_Roll"][:n]],
                "three_quaternion": [three_camera(c.model(k))["quaternion"] for k in range(n)],
            }
        cams.append(cam)
    est = np.array([[float(r[k]) if r[k] else np.nan for k in ("Loc_X", "Loc_Y", "Loc_Z")] for r in diag])
    est3 = ue_to_three(est)
    out = {
        "format": "peleng-offline-viewer/1",
        "fps": ds.fps, "n_frames": n,
        "coords": {"ue": "Unreal: см, левая, X вперёд, Y вправо, Z вверх",
                   "three": "Three.js: м, правая, Y вверх; (x,y,z) = (X_ue, Z_ue, Y_ue)/100"},
        "cameras": cams,
        "estimate": {
            "ue_cm": [[_r(v) for v in p] for p in est],
            "three_m": [[_r(v, 3) for v in p] for p in est3],
            "mode": [r["mode"] for r in diag],
            "n_cams": [int(r["n_cams"]) for r in diag],
            "cams": [[int(x) for x in r["cams"].split(",") if x] for r in diag],
            "sigma_m": [_r(float(r["sigma_m"]), 3) for r in diag],
            "speed_mps": [_r(float(r["speed_mps"]), 2) for r in diag],
        },
        "run": {k: info.get(k) for k in ("args", "git", "time_s", "metrics_summary", "trajectory_sha256")},
    }
    if ds.truth_ue is not None:
        T = ds.truth_ue[:n]
        out["truth"] = {"ue_cm": [[_r(v) for v in p] for p in T], "three_m": [[_r(v, 3) for v in p] for p in ue_to_three(T)]}
        err = np.linalg.norm(est - T, axis=1) / 100.0
        out["estimate"]["err3d_m"] = [_r(v, 3) for v in err]
    # детекции из кэша прогона
    cache = (info.get("detection") or {}).get("cache")
    if cache and Path(cache).exists():
        d = json.loads(Path(cache).read_text())
        ids = d.get("cams", [c.cam_id for c in ds.cameras])
        out["detections"] = {
            "fields": ["x1", "y1", "x2", "y2", "u", "v", "score", "cls", "source"],
            "cams": ids,
            "frames": d["frames"][:n],  # frames[k][i] — список детекций камеры cams[i] на кадре k
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="экспорт JSON для фронтенда")
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True, help="папка результатов run.py")
    ap.add_argument("--out", default=None, help="по умолчанию <run>/viewer.json")
    a = ap.parse_args(argv)
    res = build(a.data, a.run)
    p = Path(a.out) if a.out else Path(a.run) / "viewer.json"
    p.write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{p}  ({p.stat().st_size / 1e6:.1f} МБ)")


if __name__ == "__main__":
    main()
