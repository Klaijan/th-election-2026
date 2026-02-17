from __future__ import annotations

import logging
from typing import Any, List, Tuple

import cv2
import numpy as np

from . import config
from .types import DottedLine

log = logging.getLogger("extract_handwritten_numbers")


class DotDetector:
    """
    Detect dotted horizontal lines inside a given vertical zone.

    Strategy:
    - Find individual dot blobs using connected components (size + circularity filtering)
    - Group dot centers into horizontal sequences based on y proximity and x spacing
    """

    def __init__(self) -> None:
        pass

    def detect_dotted_lines(self, image: np.ndarray, zone: Tuple[int, int]) -> List[DottedLine]:
        y0, y1 = int(zone[0]), int(zone[1])
        h, w = image.shape[:2]
        y0 = max(0, min(y0, h))
        y1 = max(0, min(y1, h))
        if y1 <= y0:
            return []

        roi = image[y0:y1, :]
        dots = self._detect_dots(roi)
        # Convert ROI coords -> page coords
        dots = [(int(x), int(y + y0)) for (x, y) in dots]
        lines = self._group_into_lines(dots)
        lines.sort(key=lambda d: int(d.y))
        return lines

    def visualize_detection(self, image: np.ndarray, dotted_lines: List[Any], zone: Tuple[int, int]) -> np.ndarray:
        """
        Draw green overlays for detected dotted lines.
        """
        overlay = image.copy()
        y0, y1 = int(zone[0]), int(zone[1])
        cv2.rectangle(overlay, (0, y0), (overlay.shape[1] - 1, y1), (255, 0, 0), 2)

        for dl in dotted_lines:
            if isinstance(dl, DottedLine):
                y = int(dl.y)
                x0, x1 = int(dl.x_start), int(dl.x_end)
                dots = dl.dots
            else:
                y = int(dl["y"])
                x0, x1 = int(dl["x_start"]), int(dl["x_end"])
                dots = dl.get("dots", [])

            cv2.line(overlay, (x0, y), (x1, y), (0, 255, 0), 2, lineType=cv2.LINE_AA)
            for x, yy in dots[:2000]:
                cv2.circle(overlay, (int(x), int(yy)), 2, (0, 255, 0), -1)

        return overlay

    def _detect_dots(self, roi_bgr: np.ndarray) -> List[tuple[int, int]]:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Dots are dark on light background → invert binary makes them white.
        _thr, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Connected components on binary image
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
        if n <= 1:
            return []

        out: List[tuple[int, int]] = []
        min_d = int(config.DOT_MIN_DIAMETER)
        max_d = int(config.DOT_MAX_DIAMETER)
        min_area = int(np.pi * (min_d / 2.0) ** 2)
        max_area = int(np.pi * (max_d / 2.0) ** 2) * 3  # tolerance for slight blur

        for i in range(1, int(n)):  # 0 is background
            x, y, w, h, area = stats[i]
            if area < min_area or area > max_area:
                continue
            if w < min_d or h < min_d or w > max_d * 2 or h > max_d * 2:
                continue

            # Circularity filter: 4πA / P^2
            mask = (labels[y : y + h, x : x + w] == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            c = max(contours, key=cv2.contourArea)
            per = float(cv2.arcLength(c, True))
            if per <= 0:
                continue
            circ = float(4.0 * np.pi * float(area) / (per * per))
            if circ < float(config.DOT_CIRCULARITY_MIN):
                continue

            cx, cy = centroids[i]
            out.append((int(round(cx)), int(round(cy))))

        return out

    def _group_into_lines(self, dots: List[tuple[int, int]]) -> List[DottedLine]:
        if not dots:
            return []

        dots_sorted = sorted(dots, key=lambda p: (p[1], p[0]))
        y_tol = int(config.DOT_Y_CLUSTER_TOLERANCE_PX)

        # 1) cluster by y
        clusters: List[List[tuple[int, int]]] = []
        cur: List[tuple[int, int]] = []
        cur_y = None
        for x, y in dots_sorted:
            if cur_y is None:
                cur = [(x, y)]
                cur_y = float(y)
                continue
            if abs(float(y) - float(cur_y)) <= y_tol:
                cur.append((x, y))
                cur_y = (cur_y * (len(cur) - 1) + float(y)) / float(len(cur))
            else:
                clusters.append(cur)
                cur = [(x, y)]
                cur_y = float(y)
        if cur:
            clusters.append(cur)

        # 2) within each cluster, group by spacing sequences
        results: List[DottedLine] = []
        s_min = int(config.DOT_SPACING_MIN)
        s_max = int(config.DOT_SPACING_MAX)
        allow_gap = int(round(float(s_max) * float(config.DOT_ALLOW_GAP_MULTIPLIER)))
        min_seq = int(config.DOT_SEQUENCE_MIN)
        min_len = int(config.DOT_LINE_MIN_LENGTH)
        max_gaps = int(config.DOT_MAX_GAPS_PER_LINE)

        for cl in clusters:
            if len(cl) < min_seq:
                continue
            xs = sorted(cl, key=lambda p: p[0])

            start = 0
            gaps = 0
            for i in range(1, len(xs)):
                dx = int(xs[i][0] - xs[i - 1][0])
                ok = s_min <= dx <= s_max
                gap_ok = (dx > s_max) and (dx <= allow_gap)
                if ok:
                    continue
                if gap_ok and gaps < max_gaps:
                    gaps += 1
                    continue

                # break sequence at i-1
                self._maybe_add_line(results, xs[start:i], min_seq=min_seq, min_len=min_len)
                start = i
                gaps = 0

            self._maybe_add_line(results, xs[start:], min_seq=min_seq, min_len=min_len)

        # De-dup near-identical lines (same y within tolerance)
        results.sort(key=lambda d: (int(d.y), int(d.x_start)))
        merged: List[DottedLine] = []
        for dl in results:
            if not merged:
                merged.append(dl)
                continue
            prev = merged[-1]
            if abs(int(dl.y) - int(prev.y)) <= y_tol and abs(int(dl.x_start) - int(prev.x_start)) <= 8:
                # keep the longer / denser one
                better = dl if (dl.length, dl.dot_count) > (prev.length, prev.dot_count) else prev
                merged[-1] = better
            else:
                merged.append(dl)
        return merged

    @staticmethod
    def _maybe_add_line(out: List[DottedLine], seq: List[tuple[int, int]], *, min_seq: int, min_len: int) -> None:
        if len(seq) < int(min_seq):
            return
        xs = [p[0] for p in seq]
        ys = [p[1] for p in seq]
        x0, x1 = int(min(xs)), int(max(xs))
        y_avg = int(round(float(sum(ys)) / float(len(ys))))
        length = int(x1 - x0)
        if length < int(min_len):
            return
        out.append(
            DottedLine(
                y=y_avg,
                x_start=x0,
                x_end=x1,
                length=length,
                dot_count=int(len(seq)),
                dots=[(int(x), int(y)) for (x, y) in seq],
            )
        )


