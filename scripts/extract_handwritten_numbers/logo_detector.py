from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import config

log = logging.getLogger("extract_handwritten_numbers")


class LogoDetector:
    def __init__(self) -> None:
        self.templates_dir = Path(__file__).resolve().parent / "templates"
        self.template_path = self.templates_dir / str(getattr(config, "LOGO_TEMPLATE_FILE", "logo.png"))
        self.threshold = float(getattr(config, "LOGO_DETECTION_THRESHOLD", 0.70))
        self.scales = tuple(float(s) for s in getattr(config, "LOGO_SCALES", (1.0,)))
        self.search_y_frac = float(getattr(config, "LOGO_SEARCH_Y_FRAC", 0.20))
        self._template = cv2.imread(str(self.template_path), cv2.IMREAD_GRAYSCALE)
        if self._template is None or self._template.size == 0:
            self._template = None
            log.debug("Logo template not found or unreadable: %s", self.template_path)

    def detect_logo(self, page_bgr: np.ndarray) -> dict[str, Any]:
        default = {
            "has_logo": False,
            "confidence": 0.0,
            "position": (0, 0),
            "bbox": (0, 0, 0, 0),
            "scale": 1.0,
        }
        if self._template is None:
            return default
        if page_bgr is None or page_bgr.size == 0:
            return default

        gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        y_end = max(1, min(int(round(float(h) * self.search_y_frac)), h))
        roi = gray[:y_end, :]
        if roi.size == 0:
            return default

        best_score = -1.0
        best_bbox = (0, 0, 0, 0)
        best_scale = 1.0

        for scale in self.scales:
            tw = max(1, int(round(self._template.shape[1] * float(scale))))
            th = max(1, int(round(self._template.shape[0] * float(scale))))
            if tw > w or th > y_end:
                continue
            if tw < 3 or th < 3:
                continue

            tpl = cv2.resize(self._template, (tw, th), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
            res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if float(max_val) > best_score:
                x, y = int(max_loc[0]), int(max_loc[1])
                best_score = float(max_val)
                best_bbox = (x, y, tw, th)
                best_scale = float(scale)

        if best_score < self.threshold:
            default["confidence"] = max(0.0, float(best_score))
            return default

        x, y, tw, th = best_bbox
        return {
            "has_logo": True,
            "confidence": float(best_score),
            "position": (int(x), int(y)),
            "bbox": (int(x), int(y), int(tw), int(th)),
            "scale": float(best_scale),
        }
