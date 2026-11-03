"""Детектор дрона по шаблону (базовый, работает «из коробки» на тестовых видео).

Дрон на тестовом наборе — один и тот же объект (камера 5, кадры 0..~625, висит неподвижно в мире ≈(1330,214) px,
~73x35 px). Шаблон (RGB-патч + альфа-маска) вырезан из усреднённых кадров и лежит в assets/drone_patch.png
(перегенерация: drone_synth.extract_drone_patch).

Метод: маскированная нормированная корреляция (NCC без среднего, веса = альфа дрона) по серому изображению,
несколько масштабов шаблона (дрон может быть дальше/ближе), поиск на уменьшенном вдвое кадре; для масштабов,
где объект в кадре ≥ ~30 px, результат уточняется на полном разрешении. Причинно: кадр обрабатывается независимо
(история не нужна). Лучший пик на камеру (один дрон), порог score — NCC.

Ограничения: шаблон знает ровно этот дрон (вид сбоку/слегка сверху); на другом виде/освещении корреляция
падает — для обобщения см. TrainedDrone (trained.py, обучение на синтетике с аугментацией).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..detect_base import Detection, Detector

ASSET = Path(__file__).with_name("assets") / "drone_patch.png"


def masked_ncc(img: np.ndarray, tpl: np.ndarray, w: np.ndarray, min_std: float = 2.5) -> np.ndarray:
    """NCC изображения с шаблоном под весами w (карта размера (H-h+1, W-w+1)), float32."""
    w = w.astype(np.float32)
    sw = float(w.sum())
    tm = float((w * tpl).sum() / sw)
    tz = (tpl - tm) * w
    tstd = float(np.sqrt(((tpl - tm) ** 2 * w).sum() / sw))
    h, ww = tpl.shape
    img = img.astype(np.float64)
    img = img - img.mean()
    tz = tz.astype(np.float64)
    w = w.astype(np.float64)
    # корреляция: cv2.filter2D делает корреляцию с якорем в центре ядра — обрезаем края до «valid»
    def valid(a):
        y0, x0 = h // 2, ww // 2
        return a[y0:y0 + img.shape[0] - h + 1, x0:x0 + img.shape[1] - ww + 1]
    num = valid(cv2.filter2D(img, -1, tz, borderType=cv2.BORDER_CONSTANT))
    s1 = valid(cv2.filter2D(img, -1, w, borderType=cv2.BORDER_CONSTANT))
    s2 = valid(cv2.filter2D(img * img, -1, w, borderType=cv2.BORDER_CONSTANT))
    var = np.maximum(s2 / sw - (s1 / sw) ** 2, 0.0)
    out = num / (sw * np.sqrt(var + 1e-9) * max(tstd, 1e-6))
    out[var < min_std ** 2] = -1.0  # плоские области (небо, гладкая гора) не считаем совпадением
    return out.astype(np.float32)


class DroneTemplateDetector(Detector):
    name = "drone_tpl"

    def __init__(self, scales=(0.6, 0.8, 1.0, 1.3), thresh: float = 0.62, asset: str | None = None):
        a = cv2.imread(str(asset or ASSET), cv2.IMREAD_UNCHANGED)
        if a is None:
            raise FileNotFoundError(asset or ASSET)
        self.gray = cv2.cvtColor(a[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.alpha = (a[..., 3].astype(np.float32) / 255.0)
        self.scales = tuple(scales)
        self.thresh = thresh

    def config(self) -> dict:
        return {"name": self.name, "scales": list(self.scales), "thresh": self.thresh}

    def _tpl(self, s: float):
        h, w = self.gray.shape
        nw, nh = max(5, int(round(w * s))), max(5, int(round(h * s)))
        t = cv2.resize(self.gray, (nw, nh), interpolation=cv2.INTER_AREA)
        a = cv2.resize(self.alpha, (nw, nh), interpolation=cv2.INTER_AREA)
        return t, np.clip(a, 0, 1)

    def detect(self, cam_id, frame_idx, image, camera_model) -> list[Detection]:
        g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        half = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)
        best = None
        for s in self.scales:
            t, a = self._tpl(s * 0.5)  # поиск на половинном разрешении
            if min(t.shape) < 5 or t.shape[0] >= half.shape[0] or t.shape[1] >= half.shape[1]:
                continue
            R = masked_ncc(half, t, a)
            _, mx, _, loc = cv2.minMaxLoc(R)
            if best is None or mx > best[0]:
                best = (float(mx), s, loc)
        if best is None or best[0] < self.thresh:
            return []
        sc, s, loc = best
        t, a = self._tpl(s)
        x0, y0 = loc[0] * 2, loc[1] * 2
        # уточнение на полном разрешении в окне ±4 px
        pad = 4
        X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
        win = g[Y0:y0 + t.shape[0] + pad, X0:x0 + t.shape[1] + pad]
        if win.shape[0] > t.shape[0] and win.shape[1] > t.shape[1]:
            R = masked_ncc(win, t, a)
            _, mx, _, l2 = cv2.minMaxLoc(R)
            x0, y0, sc = X0 + l2[0], Y0 + l2[1], float(mx)
        # рамка по маске (alpha>0.3)
        ys, xs = np.nonzero(a > 0.3)
        bb = (x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1)
        pt = ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)
        return [Detection(tuple(float(v) for v in bb), pt, float(min(1.0, max(0.0, sc))), "drone", "drone_tpl", {"scale": s})]
