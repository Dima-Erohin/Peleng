"""ROI-initialized image tracking, independent of semantic detector classes.

OpenCV normalized correlation against a fixed seed prevents template drift.
Low-confidence frames carry no bounding box; this is 2D evidence, not 3D.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageMatch:
    bbox: tuple[int, int, int, int] | None
    correlation: float
    peak_margin: float
    processing_ms: float


class TemplateTracker:
    """Track a manually identified object locally at several scales.

    The seed is never replaced with an unverified detection. After a short
    disappearance search remains bounded; long losses require a new seed.
    Correlation is a similarity score, not a calibrated probability.
    """

    def __init__(
        self, image: np.ndarray, bbox: tuple[int, int, int, int],
        threshold: float = 0.65, search_radius: int = 70,
        max_lost: int = 30, peak_margin: float = 0.04,
    ):
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        if search_radius < 1 or max_lost < 0 or not 0 <= peak_margin < 1:
            raise ValueError("Invalid search radius, loss limit or peak margin")
        gray = self._gray(image)
        x, y, w, h = bbox
        if min(x, y) < 0 or min(w, h) < 8 or x + w > gray.shape[1] or y + h > gray.shape[0]:
            raise ValueError("Seed ROI must be inside the image and at least 8x8")
        self.template = gray[y:y + h, x:x + w].copy()
        if float(self.template.std()) < 4:
            raise ValueError("Seed ROI has insufficient contrast")
        self.bbox = bbox
        self.shape = gray.shape
        self.threshold = threshold
        self.radius = search_radius
        self.max_lost = max_lost
        self.min_margin = peak_margin
        self.lost = 0

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Empty frame")
        if image.dtype != np.uint8 or image.ndim not in (2, 3):
            raise ValueError("Expected uint8 grayscale or BGR frame")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    def update(self, image: np.ndarray) -> ImageMatch:
        started = time.perf_counter()
        gray = self._gray(image)
        if gray.shape != self.shape:
            raise ValueError("Frame dimensions changed")
        if self.lost > self.max_lost:
            return ImageMatch(None, 0.0, 0.0, (time.perf_counter() - started) * 1000)
        x, y, w, h = self.bbox
        radius = self.radius + min(self.lost * 3, self.radius)
        left, top = max(0, x - radius), max(0, y - radius)
        right, bottom = min(gray.shape[1], x + w + radius), min(gray.shape[0], y + h + radius)
        search = gray[top:bottom, left:right]
        best_score, best_margin, best_box = -1.0, 0.0, None
        for scale in (0.85, 0.925, 1.0, 1.075, 1.15):
            tw = round(self.template.shape[1] * scale)
            th = round(self.template.shape[0] * scale)
            if tw > search.shape[1] or th > search.shape[0]:
                continue
            template = cv2.resize(self.template, (tw, th), interpolation=cv2.INTER_LINEAR)
            response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, (px, py) = cv2.minMaxLoc(response)
            # Compare distinct peaks, not the neighboring pixels of the same peak.
            response[max(0, py - th // 2):py + th // 2 + 1,
                     max(0, px - tw // 2):px + tw // 2 + 1] = -1
            second = float(response.max())
            margin = float(score) - second
            if score > best_score:
                best_score, best_margin = float(score), margin
                best_box = (left + px, top + py, tw, th)
        if best_score >= self.threshold and best_margin >= self.min_margin and best_box is not None:
            self.bbox = best_box
            self.lost = 0
            box = best_box
        else:
            self.lost += 1
            box = None
        return ImageMatch(box, best_score, best_margin, (time.perf_counter() - started) * 1000)
