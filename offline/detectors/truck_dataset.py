"""Автоматическая разметка обучающей выборки YOLO-pose для грузовика по 5 видео + эталонной траектории.

Разметка: keypoint 0 = проекция эталонного pivot (Loc из CSV грузовика) моделью камеры кадра k; рамка = проекция
3D-параллелепипеда вокруг pivot (курс — из производной траектории, ±4 кадра; габариты ниже подобраны по масштабу
сегментационных рамок yolo11x-seg и проверены визуально: pivot — у переда кабины, L≈22 м). Это ТОЛЬКО разметка для
обучения (используются будущие кадры траектории — допустимо). На инференсе траектория не нужна.

Отбор кадров: pivot внутри кадра; грузовик не скрыт целиком — грубая проверка по причинному вычитанию фона
(пятно движения пересекает рамку), иначе кадр пропускается (гора/дерево закрывают грузовик полностью). Частичное
перекрытие остаётся (keypoint видимый, v=2): модель учится предсказывать pivot и при перекрытии.

Аугментации офлайн (только в папке aug/): вставка «листвы» (кропы деревьев из тех же видео) поверх грузовика
(имитация деревьев/кустов), размытие, шум, JPEG, яркость/контраст/тон. Остальное (масштаб 0.3–1.7, повороты, перспектива,
mosaic, HSV, отражение) — онлайн в ultralytics (см. train_truck_pose.py). В clean/ — те же кадры без аугментаций (для val).

Выход: <out>/{clean,aug}/{images,labels}/c{cam}_f{k:04d}.{jpg,txt} (картинки 1280x720), списки
<out>/lists/{trainA,valA,trainB,valB,trainAll,valAll}.txt и data_*.yaml. Эксперименты:
A: train кам.1–4 → val кам.5; B: train кам.2–5 → val кам.1; All: все камеры (val — каждый 10-й кадр, только для контроля).

Запуск: python -I offline/detectors/truck_dataset.py --data <папка с mp4+csv> --out <папка> [--stride 2]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from offline.basic_detectors import BackgroundDetector  # noqa: E402
from offline.ue_data import SyncVideoReader, load_dataset  # noqa: E402

# габариты (м) вокруг pivot в системе (вдоль курса, поперёк, вверх)
FRONT, REAR, HALF_W, HEIGHT, Z_BOTTOM = 0.3, 22.0, 1.4, 4.5, -1.7
OUT_W, OUT_H = 1280, 720


def heading(T, k):
    a = T[min(k + 4, len(T) - 1)] - T[max(k - 4, 0)]
    h = np.array([a[0], a[1]])
    n = np.linalg.norm(h)
    return h / n if n > 1e-6 else None


def truck_corners(T, k):
    hd = heading(T, k)
    if hd is None:
        return None
    nrm = np.array([-hd[1], hd[0]])
    X = []
    for s in (FRONT, -REAR):
        for t in (HALF_W, -HALF_W):
            for z in (Z_BOTTOM, Z_BOTTOM + HEIGHT):
                X.append(T[k] + np.array([hd[0] * s + nrm[0] * t, hd[1] * s + nrm[1] * t, z]))
    return np.array(X)


def label_for(T, k, cam_model, size=(1920, 1080)):
    """(bbox xyxy px, pivot px) или None, если pivot вне кадра/за камерой или рамка почти вся вне кадра."""
    W, H = size
    X = truck_corners(T, k)
    if X is None or (cam_model.depth(X) <= 0.5).any() or cam_model.depth(T[k][None])[0] <= 0.5:
        return None
    P = cam_model.project(X)
    pv = cam_model.project(T[k][None])[0]
    if not (2 <= pv[0] <= W - 2 and 2 <= pv[1] <= H - 2):
        return None
    b = np.array([P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()])
    full = (b[2] - b[0]) * (b[3] - b[1])
    b = np.array([max(b[0], 0), max(b[1], 0), min(b[2], W), min(b[3], H)])
    if (b[2] - b[0]) < 12 or (b[3] - b[1]) < 6 or (b[2] - b[0]) * (b[3] - b[1]) < 0.4 * full:
        return None
    return b, pv


def make_foliage_library(frames, rng, n=60):
    """Кропы листвы: окна с высокой долей зелёного -> (crop BGR, alpha)."""
    lib = []
    for im in frames:
        H, W = im.shape[:2]
        b, g, r = [im[..., i].astype(np.int16) for i in range(3)]
        green = ((g > r + 6) & (g > b + 6)).astype(np.uint8)
        for _ in range(60):
            s = rng.choice([80, 120, 180, 260])
            x, y = rng.randint(0, W - s), rng.randint(0, H - s)
            if green[y:y + s, x:x + s].mean() > 0.5:
                a = cv2.GaussianBlur(cv2.morphologyEx(green[y:y + s, x:x + s], cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 2.0)
                lib.append((im[y:y + s, x:x + s].copy(), np.clip(a * 1.2, 0, 1)))
                break
        if len(lib) >= n:
            break
    return lib


def add_occluders(img, box, lib, rng, pivot):
    H, W = img.shape[:2]
    n = rng.choice([1, 1, 2, 3])
    bw, bh = box[2] - box[0], box[3] - box[1]
    for _ in range(n):
        crop, a = rng.choice(lib)
        sc = rng.uniform(0.4, 1.4) * max(bw, 60) / 200.0
        w = max(12, int(crop.shape[1] * sc))
        h = max(12, int(crop.shape[0] * sc))
        c = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
        al = cv2.resize(a, (w, h))[..., None]
        # центр — внутри рамки грузовика (в 25 % случаев — на pivot)
        if rng.random() < 0.25:
            cx, cy = pivot[0] + rng.uniform(-0.3, 0.3) * w, pivot[1] + rng.uniform(-0.3, 0.3) * h
        else:
            cx, cy = rng.uniform(box[0], box[2]), rng.uniform(box[1], box[3])
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
        if xb <= xa or yb <= ya:
            continue
        cs = c[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.float32)
        als = al[ya - y0:yb - y0, xa - x0:xb - x0]
        img[ya:yb, xa:xb] = (img[ya:yb, xa:xb] * (1 - als) + cs * als).astype(np.uint8)
    return img


def photometric(img, rng):
    f = img.astype(np.float32)
    f = f * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)
    f *= np.array([rng.uniform(0.9, 1.1) for _ in range(3)], np.float32)
    if rng.random() < 0.4:
        k = rng.choice([3, 5, 7])
        f = cv2.GaussianBlur(f, (k, k), 0)
    if rng.random() < 0.3:  # размытие движения
        L = rng.randint(5, 25)
        ker = np.zeros((L, L), np.float32)
        ker[L // 2, :] = 1.0 / L
        M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), rng.uniform(-20, 20), 1.0)
        f = cv2.filter2D(f, -1, cv2.warpAffine(ker, M, (L, L)) / max(cv2.warpAffine(ker, M, (L, L)).sum(), 1e-6))
    if rng.random() < 0.5:
        f += np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(1, 6), f.shape)
    out = np.clip(f, 0, 255).astype(np.uint8)
    if rng.random() < 0.7:
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(40, 95)])
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def write_sample(root, split, name, img, box, pv, size=(1920, 1080)):
    W, H = size
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / split / "images" / f"{name}.jpg"), cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    cx, cy, w, h = (box[0] + box[2]) / 2 / W, (box[1] + box[3]) / 2 / H, (box[2] - box[0]) / W, (box[3] - box[1]) / H
    (root / split / "labels" / f"{name}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {pv[0] / W:.6f} {pv[1] / H:.6f} 2\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    ds = load_dataset(a.data)
    T = ds.truth_world()
    if T is None:
        raise SystemExit("нет эталонной траектории грузовика в папке данных")
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    bgs = {c.cam_id: BackgroundDetector() for c in ds.cameras}
    vids = [c.video_path for c in ds.cameras]
    lib_frames, lib = [], None
    kept = {c.cam_id: [] for c in ds.cameras}
    pending = []
    for k, imgs in SyncVideoReader(vids, ds.n_frames):
        for cam, im in zip(ds.cameras, imgs):
            if im is None:
                continue
            blobs = bgs[cam.cam_id].detect(cam.cam_id, k, im, cam.model(k))  # каждый кадр: копит фон
            if k % 97 == 5 and len(lib_frames) < 40:
                lib_frames.append(im.copy())
            if k % a.stride or k < 12:
                continue
            lab = label_for(T, k, cam.model(k), cam.size)
            if lab is None:
                continue
            box, pv = lab
            hit = any(b.bbox[0] < box[2] and b.bbox[2] > box[0] and b.bbox[1] < box[3] and b.bbox[3] > box[1] for b in blobs)
            if not hit:
                continue
            pending.append((cam.cam_id, k, im.copy(), box, pv))
        if len(pending) >= 40 or (k == ds.n_frames - 1):
            if lib is None and len(lib_frames) >= 12:
                lib = make_foliage_library(lib_frames, rng)
                print("кропов листвы:", len(lib))
            for cam_id, kk, im, box, pv in pending:
                name = f"c{cam_id}_f{kk:04d}"
                write_sample(root, "clean", name, im, box, pv)
                aug = im.copy()
                if lib and rng.random() < 0.45:
                    aug = add_occluders(aug, box, lib, rng, pv)
                aug = photometric(aug, rng)
                write_sample(root, "aug", name, aug, box, pv)
                kept[cam_id].append(name)
            pending = []
        if k % 200 == 0:
            print(k, {c: len(v) for c, v in kept.items()}, flush=True)
    lists = root / "lists"
    lists.mkdir(exist_ok=True)

    def paths(split, cams, every=1):
        out = []
        for c in cams:
            for i, n in enumerate(sorted(kept[c])):
                if i % every == 0:
                    out.append((root / split / "images" / f"{n}.jpg").as_posix())
        return out

    exps = {"A": ([1, 2, 3, 4], [5]), "B": ([2, 3, 4, 5], [1]), "All": ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])}
    for e, (tr, va) in exps.items():
        (lists / f"train{e}.txt").write_text("\n".join(paths("aug", tr)) + "\n")
        (lists / f"val{e}.txt").write_text("\n".join(paths("clean", va, every=(10 if e == "All" else 2))) + "\n")
        (root / f"data_{e}.yaml").write_text(
            f"path: {root.as_posix()}\ntrain: {(lists / f'train{e}.txt').as_posix()}\nval: {(lists / f'val{e}.txt').as_posix()}\n"
            f"kpt_shape: [1, 3]\nflip_idx: [0]\nnames:\n  0: truck\n")
    print("готово", {c: len(v) for c, v in kept.items()})


if __name__ == "__main__":
    main()
