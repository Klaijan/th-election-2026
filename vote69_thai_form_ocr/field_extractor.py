from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from . import config
from .dot_detector import DotDetector
from .types import Box, DottedLine, ExtractedRegion

log = logging.getLogger("vote69_form_ocr")


class FieldExtractor:
    """
    Extract handwriting regions around dotted lines in the "fields" zone (page 1).
    """

    def __init__(self, *, remove_dots: bool = True) -> None:
        self.remove_dots = bool(remove_dots)
        self.dot_detector = DotDetector()

    def extract_fields(self, image: np.ndarray, zone: Tuple[int, int], *, debug: bool = False) -> List[ExtractedRegion]:
        dotted_lines = self.dot_detector.detect_dotted_lines(image, zone)
        if debug:
            log.info("Detected %d dotted line(s) in zone-1 fields range", len(dotted_lines))
        return self.extract_field_regions(image, dotted_lines)

    # Backwards-compat alias (older code/tests may still call this).
    def extract_from_zone(self, image: np.ndarray, zone: Tuple[int, int], *, debug: bool = False) -> List[ExtractedRegion]:
        return self.extract_fields(image, zone, debug=debug)

    def extract_field_regions(self, image: np.ndarray, dotted_lines: List[Any]) -> List[ExtractedRegion]:
        h, w = image.shape[:2]
        out: List[ExtractedRegion] = []

        for i, dl_any in enumerate(dotted_lines):
            dl: Dict[str, Any]
            if isinstance(dl_any, DottedLine):
                dl = {
                    "y": dl_any.y,
                    "x_start": dl_any.x_start,
                    "x_end": dl_any.x_end,
                    "dot_count": dl_any.dot_count,
                    "dots": dl_any.dots,
                }
            else:
                dl = dl_any

            y = int(dl["y"])
            x0 = int(dl["x_start"])
            x1 = int(dl["x_end"])

            pad_top = int(config.EXTRACTION_PAD_TOP)
            pad_bottom = int(config.EXTRACTION_PAD_BOTTOM)
            height = int(pad_top + pad_bottom)
            box = Box(
                x=x0 - int(config.EXTRACTION_PAD_LEFT_RIGHT),
                y=y - int(pad_top),
                w=(x1 - x0) + int(config.EXTRACTION_PAD_LEFT_RIGHT) * 2,
                h=int(height),
            ).clamp(width=w, height=h)

            raw = image[box.y : box.y2, box.x : box.x2].copy()
            if raw.size == 0:
                continue

            pre = self._preprocess_region(raw)
            if self._is_empty(pre):
                continue

            out.append(
                ExtractedRegion(
                    region_id=f"field_{i+1}",
                    bbox=box,
                    image=pre,
                    raw_image=raw,
                    meta={"dot_line": dl},
                )
            )

        return out

    def _preprocess_region(self, bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        if self.remove_dots:
            gray = self._remove_dot_artifacts(gray)

        # Otsu binarization (black ink as 0, background 255)
        _thr, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Light denoise
        bw = cv2.fastNlMeansDenoising(bw, None, h=10, templateWindowSize=7, searchWindowSize=21)
        return bw

    def _remove_dot_artifacts(self, gray: np.ndarray) -> np.ndarray:
        """
        Best-effort removal of tiny circular blobs (dotted line), to reduce OCR distraction.
        """
        inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)
        if n <= 1:
            return gray

        out = gray.copy()
        min_d = int(config.DOT_MIN_DIAMETER)
        max_d = int(config.DOT_MAX_DIAMETER)
        min_area = int(np.pi * (min_d / 2.0) ** 2)
        max_area = int(np.pi * (max_d / 2.0) ** 2) * 4
        for i in range(1, int(n)):
            x, y, w, h, area = stats[i]
            if area < min_area or area > max_area:
                continue
            if w > max_d * 2 or h > max_d * 2:
                continue
            cx, cy = centroids[i]
            r = int(max(2, round(max(w, h) / 2.0) + 1))
            cv2.circle(out, (int(round(cx)), int(round(cy))), r, 255, -1)
        return out

    @staticmethod
    def _is_empty(bw: np.ndarray) -> bool:
        if bw.size == 0:
            return True
        # consider "ink" as dark pixels
        ink = int(np.sum(bw < 200))
        return ink < 30


