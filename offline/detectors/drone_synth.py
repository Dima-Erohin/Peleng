"""Генерация синтетического датасета для дрона (формат YOLO) из реальных кадров тестового набора.

Идея
----
1. Дрон (камера 5, кадры 0..565 висит на месте ≈(1330,213) px, ~73x35 px) вырезается из усреднённого по
   времени кадра: усреднение подавляет шум сжатия, альфа-маска = отличие от фона, восстановленного
   инпейнтингом вокруг дрона.
2. Фоны — случайные окна из кадров всех 5 видео (горы, деревья, небо, земля); область самого дрона в
   камере 5 на кадрах 0..565 исключается из фонов.
3. Вставка с аугментацией: масштаб ширины дрона 8..120 px, поворот, отражение, яркость/контраст/гамма/тон,
   размытие движения, гауссово размытие, шум, JPEG-сжатие. ~15% картинок — без дрона (негативы).
4. Разбиение train/val — ПО КАДРАМ-ИСТОЧНИКАМ фона (val-фоны не видны в train).

Запуск (пути — аргументами):
    python -I -m offline.detectors.drone_synth --data <папка с mp4+csv> --out <папка датасета> \
        [--n-train 6000 --n-val 600 --imgsz 640 --seed 0]

Результат: <out>/images/{train,val}, <out>/labels/{train,val}, <out>/data.yaml, <out>/drone_patch.png.
Обучение: offline/detectors/train_drone.py.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

# bbox дрона на кадре камеры 5 (кадры 0..565), полученный по наблюдению
DRONE_BOX = (1293, 195, 1367, 231)
DRONE_FRAMES = 566


def extract_drone_patch(video5: str, pad: int = 12):
    """Возвращает (rgb-патч float32 HxWx3, альфа float32 HxW 0..1) дрона."""
    cap = cv2.VideoCapture(video5)
    x0, y0, x1, y1 = DRONE_BOX
    X0, Y0, X1, Y1 = x0 - pad - 20, y0 - pad - 20, x1 + pad + 20, y1 + pad + 20
    acc = None
    n = 0
    for _ in range(DRONE_FRAMES):
        ok, im = cap.read()
        if not ok:
            break
        p = im[Y0:Y1, X0:X1].astype(np.float32)
        acc = p if acc is None else acc + p
        n += 1
    avg = acc / n
    m = np.zeros(avg.shape[:2], np.uint8)
    m[(y0 - Y0) - 3:(y1 - Y0) + 3, (x0 - X0) - 3:(x1 - X0) + 3] = 255
    bg = cv2.inpaint(avg.astype(np.uint8), m, 6, cv2.INPAINT_TELEA).astype(np.float32)
    bgs = cv2.GaussianBlur(bg, (0, 0), 4)
    diff = np.abs(avg - bgs).max(axis=2)
    alpha = np.clip((diff - 8.0) / 22.0, 0, 1)
    alpha[m == 0] = 0
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.6)
    ys, xs = np.nonzero(alpha > 0.3)
    ya, yb, xa, xb = ys.min() - 2, ys.max() + 3, xs.min() - 2, xs.max() + 3
    return avg[ya:yb, xa:xb], alpha[ya:yb, xa:xb]


def load_backgrounds(video_paths: dict[int, str], per_cam: int, rng: random.Random):
    """Список (cam, frame, bgr-кадр 1080p) — разные кадры каждой камеры."""
    out = []
    for cam, path in video_paths.items():
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = sorted(rng.sample(range(0, n), per_cam))
        for fr in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, im = cap.read()
            if ok:
                out.append((cam, fr, im))
    return out


def random_window(im, cam, fr, size, rng):
    """Случайное окно size x size (с апскейлом/даунскейлом исходника), без области реального дрона."""
    H, W = im.shape[:2]
    for _ in range(20):
        s = rng.choice([1.0, 1.0, 0.75, 0.5, 1.5])  # масштаб фона (разные «дальности»)
        win = int(size / s)
        if win > min(H, W):
            continue
        x = rng.randint(0, W - win)
        y = rng.randint(0, H - win)
        if cam == 5 and fr < DRONE_FRAMES:
            bx0, by0, bx1, by1 = DRONE_BOX
            if not (x + win < bx0 - 10 or x > bx1 + 10 or y + win < by0 - 10 or y > by1 + 10):
                continue
        crop = im[y:y + win, x:x + win]
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA if s <= 1 else cv2.INTER_LINEAR)
    return cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)


def motion_blur(img, length, angle):
    if length < 2:
        return img
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    k /= max(k.sum(), 1e-6)
    return cv2.filter2D(img, -1, k)


def paste(bg, patch, alpha, rng):
    """Вставляет патч в bg (uint8 BGR). Возвращает (img, bbox) либо (bg, None)."""
    S = bg.shape[0]
    w_t = np.exp(rng.uniform(np.log(8), np.log(120)))  # ширина дрона 8..120 px (лог-равномерно)
    sc = w_t / patch.shape[1]
    ang = rng.uniform(-35, 35)
    flip = rng.random() < 0.5
    p = patch.copy()
    a = alpha.copy()
    if flip:
        p, a = p[:, ::-1], a[:, ::-1]
    interp = cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC
    nw, nh = max(3, int(round(p.shape[1] * sc))), max(3, int(round(p.shape[0] * sc)))
    p = cv2.resize(np.ascontiguousarray(p), (nw, nh), interpolation=interp)
    a = cv2.resize(np.ascontiguousarray(a), (nw, nh), interpolation=interp)
    # поворот с расширением холста
    d = int(np.ceil(np.hypot(nw, nh)))
    cv_p = np.zeros((d, d, 3), np.float32)
    cv_a = np.zeros((d, d), np.float32)
    ox, oy = (d - nw) // 2, (d - nh) // 2
    cv_p[oy:oy + nh, ox:ox + nw] = p
    cv_a[oy:oy + nh, ox:ox + nw] = a
    M = cv2.getRotationMatrix2D((d / 2, d / 2), ang, 1.0)
    cv_p = cv2.warpAffine(cv_p, M, (d, d), flags=cv2.INTER_LINEAR)
    cv_a = cv2.warpAffine(cv_a, M, (d, d), flags=cv2.INTER_LINEAR)
    # оптические вариации «объекта»
    gain = rng.uniform(0.55, 1.25)
    tint = np.array([rng.uniform(0.9, 1.1) for _ in range(3)], np.float32)
    cv_p = np.clip(cv_p * gain * tint, 0, 255)
    # размытие (ухудшение резкости для далёкого/быстрого объекта) — применяется к объекту и альфе
    if rng.random() < 0.5:
        L = rng.randint(2, max(3, int(0.25 * w_t) + 2))
        ma = rng.uniform(0, 180)
        cv_p = motion_blur(cv_p, L, ma)
        cv_a = motion_blur(cv_a, L, ma)
    if rng.random() < 0.5:
        sg = rng.uniform(0.3, 1.3)
        cv_p = cv2.GaussianBlur(cv_p, (0, 0), sg)
        cv_a = cv2.GaussianBlur(cv_a, (0, 0), sg)
    ys, xs = np.nonzero(cv_a > 0.25)
    if len(xs) == 0:
        return bg, None
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    if d >= S:
        return bg, None
    px = rng.randint(0, S - d)
    py = rng.randint(0, S - d)
    # фон под объектом
    reg = bg[py:py + d, px:px + d].astype(np.float32)
    alpha_g = cv_a[..., None] * rng.uniform(0.75, 1.0)  # лёгкая полупрозрачность (дымка)
    out = reg * (1 - alpha_g) + cv_p * alpha_g
    img = bg.copy()
    img[py:py + d, px:px + d] = np.clip(out, 0, 255).astype(np.uint8)
    x1, y1 = px + xs.min(), py + ys.min()
    return img, (x1, y1, x1 + bw, y1 + bh)


def post_aug(img, rng):
    """Глобальные аугментации кадра (после вставки): яркость/контраст/гамма, шум, JPEG."""
    f = img.astype(np.float32)
    if rng.random() < 0.7:
        c, b = rng.uniform(0.7, 1.3), rng.uniform(-25, 25)
        f = f * c + b
    if rng.random() < 0.3:
        g = rng.uniform(0.7, 1.4)
        f = 255 * (np.clip(f, 0, 255) / 255) ** g
    if rng.random() < 0.5:
        f = f + np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(1, 6), f.shape)
    img = np.clip(f, 0, 255).astype(np.uint8)
    if rng.random() < 0.8:
        q = rng.randint(35, 95)
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


def make_split(name, n, bgs, patch, alpha, out, size, rng, neg_frac=0.15, max_per_img=2):
    (out / "images" / name).mkdir(parents=True, exist_ok=True)
    (out / "labels" / name).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cam, fr, im = rng.choice(bgs)
        bg = random_window(im, cam, fr, size, rng)
        labels = []
        if rng.random() > neg_frac:
            for _ in range(rng.choice([1] * 4 + [2])):
                bg, box = paste(bg, patch, alpha, rng)
                if box is not None:
                    x1, y1, x2, y2 = box
                    labels.append(((x1 + x2) / 2 / size, (y1 + y2) / 2 / size, (x2 - x1) / size, (y2 - y1) / size))
        bg = post_aug(bg, rng)
        stem = f"{name}_{i:06d}"
        cv2.imwrite(str(out / "images" / name / f"{stem}.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(out / "labels" / name / f"{stem}.txt", "w") as f:
            for (cx, cy, w, h) in labels:
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="папка с Way1.FinalImage.CineCameraActorN.mp4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vids = {c: str(data / f"Way1.FinalImage.CineCameraActor{c}.mp4") for c in range(1, 6)}
    patch, alpha = extract_drone_patch(vids[5])
    cv2.imwrite(str(out / "drone_patch.png"), np.dstack([patch.astype(np.uint8), (alpha * 255).astype(np.uint8)]))
    print("патч дрона", patch.shape)
    bgs = load_backgrounds(vids, per_cam=60, rng=rng)
    rng.shuffle(bgs)
    nval = max(5, len(bgs) // 6)
    val_bgs, train_bgs = bgs[:nval], bgs[nval:]
    print(f"фонов: train {len(train_bgs)}, val {len(val_bgs)}")
    make_split("train", a.n_train, train_bgs, patch, alpha, out, a.imgsz, rng)
    make_split("val", a.n_val, val_bgs, patch, alpha, out, a.imgsz, rng)
    (out / "data.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: drone\n")
    print("готово:", out)


if __name__ == "__main__":
    main()
