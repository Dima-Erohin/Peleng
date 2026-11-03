"""Оценка обученного YOLO-pose на списке картинок (val-список из truck_dataset.py): доля найденных, ошибка pivot в px
исходного кадра 1920x1080 (медиана/p90/среднее), мс/кадр.

    python -I offline/detectors/eval_truck_pose.py --weights best.pt --list <ds>/lists/valA.txt [--imgsz 800] [--conf 0.1]
«Найден» = есть детекция, чей pivot ближе 0.5·размер_рамки_GT к эталонному pivot; ошибка считается для лучшей по score детекции.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    from ultralytics import YOLO

    m = YOLO(a.weights)
    files = [l.strip() for l in open(a.list) if l.strip()]
    if a.max:
        files = files[:a.max]
    errs, found, any_det, ts, per = [], 0, 0, [], {}
    for f in files:
        lab = Path(f.replace("/images/", "/labels/")).with_suffix(".txt").read_text().split()
        gx, gy, gw = float(lab[5]) * 1920, float(lab[6]) * 1080, float(lab[3]) * 1920
        t = time.time()
        r = m.predict(f, imgsz=a.imgsz, conf=a.conf, verbose=False, device=a.device)[0]
        ts.append(time.time() - t)
        cam = Path(f).name[:2]
        per.setdefault(cam, [0, 0, []])
        per[cam][1] += 1
        if r.boxes is None or len(r.boxes) == 0:
            continue
        any_det += 1
        W, H = r.orig_shape[1], r.orig_shape[0]
        kp = r.keypoints.xy.cpu().numpy()  # в пикселях исходной картинки (1280x720) -> в 1920x1080
        sx, sy = 1920 / W, 1080 / H
        best = int(np.argmax(r.boxes.conf.cpu().numpy()))
        px, py = kp[best][0][0] * sx, kp[best][0][1] * sy
        e = float(np.hypot(px - gx, py - gy))
        if e < 0.5 * max(gw, 40):
            found += 1
            per[cam][0] += 1
        errs.append(e)
    errs = np.array(errs)
    print(f"картинок {len(files)}, детекция есть {any_det} ({100*any_det/len(files):.0f}%), найден (pivot в 0.5·ширины GT) {found} ({100*found/len(files):.0f}%)")
    if len(errs):
        print(f"ошибка pivot px (лучшая детекция): медиана {np.median(errs):.1f}, p90 {np.percentile(errs,90):.1f}, среднее {errs.mean():.1f}; "
              f"RMSE {np.sqrt((errs**2).mean()):.1f}")
        print("по камерам (найдено/всего):", {k: f"{v[0]}/{v[1]}" for k, v in per.items()})
    print(f"мс/кадр: {1000*np.mean(ts[1:]):.0f}")


if __name__ == "__main__":
    main()
