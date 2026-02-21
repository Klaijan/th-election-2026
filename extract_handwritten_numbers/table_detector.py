from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import config
from .types import Box, TableStructure

if TYPE_CHECKING:
    from .color_separator import InkMasks

log = logging.getLogger("extract_handwritten_numbers")


class TableDetector:
    """
    Detect a table bbox within a zone, then parse grid lines to identify column boundaries.

    Implementation is intentionally similar to `extract_tables_opencv.py` already in this repo
    (binarize → morphology → projections → merged line positions), but returns a compact
    structure suitable for multi-page "reuse column 3" extraction.
    """

    def __init__(self) -> None:
        pass

    def detect_table_boundary(
        self,
        image: np.ndarray,
        zone: Tuple[int, int],
        *,
        ink_masks: Optional["InkMasks"] = None,
    ) -> Box:
        y0, y1 = int(zone[0]), int(zone[1])
        h, w = image.shape[:2]
        y0 = max(0, min(y0, h))
        y1 = max(0, min(y1, h))
        roi = image[y0:y1, :]
        if roi.size == 0:
            return Box(x=0, y=y0, w=w, h=max(0, y1 - y0))

        if ink_masks is not None and ink_masks.has_blue:
            # Use black mask directly — grid lines are black
            bw = ink_masks.black[y0:y1, :]
        else:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            bw = self._binarize_for_lines(gray)
        vert, horiz, grid = self._detect_grid_masks(bw)

        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return Box(x=0, y=y0, w=w, h=max(0, y1 - y0))

        best = None
        best_area = 0
        for c in contours:
            x, y, ww, hh = cv2.boundingRect(c)
            area = int(ww * hh)
            if ww < 200 or hh < 200:
                continue
            if area > best_area:
                best_area = area
                best = (x, y, ww, hh)

        if best is None:
            return Box(x=0, y=y0, w=w, h=max(0, y1 - y0))

        x, y, ww, hh = best
        # pad slightly to include borders
        pad = 8
        box = Box(x=int(x - pad), y=int(y0 + y - pad), w=int(ww + pad * 2), h=int(hh + pad * 2))
        return box.clamp(width=w, height=h)

    def parse_table_structure(
        self,
        image: np.ndarray,
        table_bbox: Box,
        *,
        ink_masks: Optional["InkMasks"] = None,
    ) -> TableStructure:
        h, w = image.shape[:2]
        box = table_bbox.clamp(width=w, height=h)
        table = image[box.y : box.y2, box.x : box.x2]
        if table.size == 0:
            raise ValueError("Empty table crop")

        black_crop = None
        if ink_masks is not None and ink_masks.has_blue:
            black_crop = ink_masks.black[box.y : box.y2, box.x : box.x2]
        grid = self.detect_grid_lines(table, ink_masks_crop=black_crop)
        horiz_local = grid["horizontal"]
        vert_local = grid["vertical"]
        horiz_abs = [int(box.y + y) for y in horiz_local]
        vert_abs = [int(box.x + x) for x in vert_local]
        rows = max(0, len(horiz_abs) - 1)
        cols = max(0, len(vert_abs) - 1)
        x_range, col_idx = self.identify_target_column(vert_abs)

        return TableStructure(
            bbox=box,
            grid_horizontal=horiz_abs,
            grid_vertical=vert_abs,
            target_column=x_range,
            target_column_index=int(col_idx),
            rows=int(rows),
            cols=int(cols),
        )

    def detect_grid_lines(
        self,
        table_bgr: np.ndarray,
        *,
        ink_masks_crop: Optional[np.ndarray] = None,
    ) -> Dict[str, List[int]]:
        if ink_masks_crop is not None:
            # Pre-cropped black mask provided
            bw = ink_masks_crop
        else:
            gray = cv2.cvtColor(table_bgr, cv2.COLOR_BGR2GRAY)
            bw = self._binarize_for_lines(gray)
        vert, horiz, _grid = self._detect_grid_masks(bw)

        # Projections (how much line ink per col/row)
        vx = np.sum(vert > 0, axis=0).astype(np.float32)
        vy = np.sum(horiz > 0, axis=1).astype(np.float32)

        x_lines = self._line_positions_1d(vx, thr_frac=0.55)
        y_lines = self._line_positions_1d(vy, thr_frac=0.55)

        x_lines = self._merge_nearby(x_lines, tol=int(config.TABLE_GRID_MERGE_TOLERANCE))
        y_lines = self._merge_nearby(y_lines, tol=int(config.TABLE_GRID_MERGE_TOLERANCE))

        return {"horizontal": y_lines, "vertical": x_lines}

    def identify_target_column(self, vertical_lines_abs: List[int]) -> tuple[Tuple[int, int], int]:
        """
        Choose the column to extract. Current requirement: last column.

        Returns ((x_start, x_end), col_index_0based).
        """
        xs = sorted({int(x) for x in vertical_lines_abs})
        if len(xs) < 2:
            raise ValueError(f"Not enough vertical lines to pick a column (need >=2, got {len(xs)})")

        mode = str(getattr(config, "TABLE_TARGET_COLUMN", "last")).lower().strip()
        if mode == "last":
            return ((int(xs[-2]), int(xs[-1])), int(len(xs) - 2))
        if mode == "index":
            idx = int(getattr(config, "TABLE_TARGET_COLUMN_INDEX", 0))
            if idx < 0:
                idx = max(0, (len(xs) - 1) + idx)
            if len(xs) < idx + 2:
                raise ValueError(f"Not enough vertical lines to pick column idx={idx} (need >= {idx+2}, got {len(xs)})")
            return ((int(xs[idx]), int(xs[idx + 1])), int(idx))
        raise ValueError(f"Unsupported TABLE_TARGET_COLUMN mode: {mode}")

    @staticmethod
    def _binarize_for_lines(gray: np.ndarray) -> np.ndarray:
        inv = cv2.bitwise_not(gray)
        bw = cv2.adaptiveThreshold(inv, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, -5)
        return bw

    @staticmethod
    def _detect_grid_masks(bw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = bw.shape[:2]
        k = max(12, int(min(w, h) / 120))
        k_h = max(int(config.TABLE_MIN_HORIZ_KERNEL), int(k))
        k_v = max(int(config.TABLE_MIN_VERT_KERNEL), int(k))

        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_v))
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_h, 1))

        vert = cv2.erode(bw, vert_kernel, iterations=2)
        vert = cv2.dilate(vert, vert_kernel, iterations=2)

        horiz = cv2.erode(bw, horiz_kernel, iterations=2)
        horiz = cv2.dilate(horiz, horiz_kernel, iterations=2)

        grid = cv2.bitwise_or(vert, horiz)
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        return vert, horiz, grid

    @staticmethod
    def _line_positions_1d(proj: np.ndarray, *, thr_frac: float) -> List[int]:
        proj = np.asarray(proj, dtype=np.float32)
        if proj.size == 0:
            return []
        mx = float(proj.max()) if proj.size else 0.0
        if mx <= 0:
            return []
        thr = mx * float(thr_frac)
        on = proj >= thr

        idxs: List[int] = []
        i = 0
        n = int(on.size)
        while i < n:
            if not bool(on[i]):
                i += 1
                continue
            j = i + 1
            while j < n and bool(on[j]):
                j += 1
            center = int(round((i + (j - 1)) / 2.0))
            idxs.append(center)
            i = j
        idxs.sort()
        return idxs

    @staticmethod
    def _merge_nearby(xs: List[int], *, tol: int) -> List[int]:
        xs = sorted(int(x) for x in xs)
        if not xs:
            return []
        out: List[int] = [xs[0]]
        for x in xs[1:]:
            if abs(int(x) - int(out[-1])) <= int(tol):
                out[-1] = int(round((out[-1] + int(x)) / 2.0))
            else:
                out.append(int(x))
        return out


