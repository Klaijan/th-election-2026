from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

from . import config
from .types import Box, ExtractedRegion, TableStructure

log = logging.getLogger("vote69_form_ocr")


class TableExtractor:
    """
    Extract only **column 3** cell images from a table (multi-page continuation supported).
    """

    def __init__(self, *, remove_grid: bool = True) -> None:
        self.remove_grid = bool(remove_grid)

    def extract_target_column_cells(
        self,
        image: np.ndarray,
        table_structure: TableStructure,
        page_num: int,
        *,
        row_offset: int = 0,
        skip_header_rows: int = 1,
    ) -> List[ExtractedRegion]:
        h, w = image.shape[:2]
        (x0, x1) = table_structure.target_column
        col_idx0 = int(table_structure.target_column_index)
        col_idx1 = int(col_idx0 + 1)
        ys = list(table_structure.grid_horizontal)
        if len(ys) < 2:
            return []

        out: List[ExtractedRegion] = []
        for r in range(int(skip_header_rows), len(ys) - 1):
            y_top = int(ys[r])
            y_bot = int(ys[r + 1])
            box = Box(x=int(x0), y=int(y_top), w=int(x1 - x0), h=int(y_bot - y_top)).clamp(width=w, height=h)
            if box.w <= 0 or box.h <= 0:
                continue

            raw = image[box.y : box.y2, box.x : box.x2].copy()
            if raw.size == 0:
                continue

            # Filter: only keep cells that likely contain handwriting over dots
            if not self._cell_has_dots(raw):
                continue

            pre = self._preprocess_cell(raw)
            if self._is_empty(pre):
                continue

            row_num = int(row_offset + (r - int(skip_header_rows)) + 1)
            region_id = f"table_p{int(page_num)+1}_r{row_num}_c{col_idx1}"
            out.append(
                ExtractedRegion(
                    region_id=region_id,
                    bbox=box,
                    image=pre,
                    raw_image=raw,
                    meta={
                        "page": int(page_num) + 1,
                        "row": row_num,
                        "column_index_0based": int(col_idx0),
                        "column_index_1based": int(col_idx1),
                        "column_label": "last" if str(getattr(config, "TABLE_TARGET_COLUMN", "last")).lower().strip() == "last" else "index",
                        "has_dots": True,
                    },
                )
            )

        return out

    def _cell_has_dots(self, cell_bgr: np.ndarray) -> bool:
        """
        Fast heuristic: count small circular-ish blobs and ensure they span horizontally.
        This is cheaper/looser than full line reconstruction, and works well per-cell.
        """
        gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
        _thr, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
        if n <= 1:
            return False

        min_d = int(config.DOT_MIN_DIAMETER)
        max_d = int(config.DOT_MAX_DIAMETER)
        min_area = int(np.pi * (min_d / 2.0) ** 2)
        max_area = int(np.pi * (max_d / 2.0) ** 2) * 4

        centers: List[Tuple[int, int]] = []
        for i in range(1, int(n)):
            x, y, w, h, area = stats[i]
            if area < min_area or area > max_area:
                continue
            if w > max_d * 2 or h > max_d * 2:
                continue
            cx, cy = centroids[i]
            centers.append((int(round(cx)), int(round(cy))))

        if len(centers) < int(config.DOT_SEQUENCE_MIN):
            return False

        xs = [c[0] for c in centers]
        span = int(max(xs) - min(xs)) if xs else 0
        # table cells are narrower than full fields; accept shorter spans
        return span >= max(40, int(cell_bgr.shape[1] * 0.35))

    def _preprocess_cell(self, bgr: np.ndarray) -> np.ndarray:
        # remove borders by shaving a few px
        shave = 3
        h, w = bgr.shape[:2]
        bgr = bgr[shave : max(shave, h - shave), shave : max(shave, w - shave)].copy()

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        if self.remove_grid:
            gray = self._remove_grid_lines(gray)

        _thr, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw = cv2.fastNlMeansDenoising(bw, None, h=10, templateWindowSize=7, searchWindowSize=21)
        bw = self._crop_to_content(bw)
        return bw

    @staticmethod
    def _remove_grid_lines(gray: np.ndarray) -> np.ndarray:
        inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        h, w = inv.shape[:2]
        kx = max(18, int(w / 18))
        ky = max(18, int(h / 10))
        horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
        vert_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky))

        horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_k, iterations=1)
        vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN, vert_k, iterations=1)
        lines = cv2.bitwise_or(horiz, vert)

        out = gray.copy()
        out[lines > 0] = 255
        return out

    @staticmethod
    def _crop_to_content(bw: np.ndarray) -> np.ndarray:
        if bw.size == 0:
            return bw
        ys, xs = np.where(bw < 200)
        if len(xs) == 0 or len(ys) == 0:
            return bw
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        pad = 2
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(bw.shape[1] - 1, x1 + pad)
        y1 = min(bw.shape[0] - 1, y1 + pad)
        return bw[y0 : y1 + 1, x0 : x1 + 1].copy()

    @staticmethod
    def _is_empty(bw: np.ndarray) -> bool:
        ink = int(np.sum(bw < 200))
        return ink < 20


