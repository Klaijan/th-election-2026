from __future__ import annotations

import logging
from typing import Dict, List, Tuple
from pathlib import Path

import cv2
import numpy as np

from . import config

log = logging.getLogger("extract_handwritten_numbers")


class ZoneDetector:
    """
    Page zoning:
    - Page 1: infer the **fields (zone 1)** y-range from `template_4.png` (top) + `template_5.png` (bottom) matches.
      If not found, fall back to a broad "above table" band.
      and keep a percent-based **table** zone to guide table boundary detection.
    - Pages 2+: table-only (full page).
    """

    def __init__(self) -> None:
        self._logged_zone1_template_fallback = False

    @staticmethod
    def _zone_px(height: int, frac_zone: Tuple[float, float]) -> Tuple[int, int]:
        y0 = int(round(float(frac_zone[0]) * height))
        y1 = int(round(float(frac_zone[1]) * height))
        y0 = max(0, min(y0, height))
        y1 = max(0, min(y1, height))
        if y1 < y0:
            y0, y1 = y1, y0
        return (y0, y1)

    def classify_pages(self, pages: List[np.ndarray]) -> List[Dict]:
        out: List[Dict] = []
        for i, page in enumerate(pages):
            h, w = page.shape[:2]
            if i == 0:
                # Page 1:
                # - No header zone (template/bullet approach makes header zoning unnecessary)
                # - Fields zone inferred from zone-1 templates (template_4 = top, template_5 = bottom)
                header = (0, 0)
                table = self._zone_px(h, config.TABLE_ZONE)
                # Zone-1 templates search window is independent of table zone:
                # left X% (config.ZONE1_TEMPLATE_SEARCH_X_FRAC) and top Y% (config.ZONE1_TEMPLATE_SEARCH_Y_FRAC)
                y_end = int(round(float(h) * float(getattr(config, "ZONE1_TEMPLATE_SEARCH_Y_FRAC", 0.65))))
                y_end = max(1, min(y_end, h))
                x_end = int(round(float(w) * float(getattr(config, "ZONE1_TEMPLATE_SEARCH_X_FRAC", 0.5))))
                x_end = max(1, min(x_end, w))
                fields, zone1_hits = self._infer_fields_zone_from_zone1_templates(
                    page, search_zone=(0, int(y_end)), search_x_end=int(x_end)
                )
                if fields == (0, 0):
                    # Fallback: broad top-Y% band (still avoids table area in most forms).
                    fields = (0, int(y_end))
                    zone1_hits = []
            else:
                # Continuation pages: treat as table-only (full page).
                header = (0, 0)
                fields = (0, 0)
                table = (0, h)
                zone1_hits = []
                x_end = 0
                y_end = 0

            has_table = self._has_table_grid(page, table)
            has_fields = (i == 0)  # typical forms: only page 1 has bullet fields

            out.append(
                {
                    "page_num": i,
                    "zones": {"header": header, "fields": fields, "table": table},
                    "has_fields": bool(has_fields),
                    "has_table": bool(has_table),
                    "page_size": (w, h),
                    "zone1_template_hits": zone1_hits,
                    "zone1_template_search": {"x_end": int(x_end), "y_end": int(y_end)},
                }
            )
        return out

    def _infer_fields_zone_from_zone1_templates(
        self, page_bgr: np.ndarray, *, search_zone: Tuple[int, int], search_x_end: int
    ) -> tuple[Tuple[int, int], list[dict]]:
        """
        Use template_4 + template_5 to define the y-range for zone 1 (page 1).

        Returns:
        - (y0, y1) in absolute pixels, or (0,0) when templates are unavailable/not matched
        - hits: list of dicts for debug overlays
        """
        y0, y1 = int(search_zone[0]), int(search_zone[1])
        h, w = page_bgr.shape[:2]
        y0 = max(0, min(y0, h))
        y1 = max(0, min(y1, h))
        if y1 <= y0:
            return (0, 0), []

        xw = max(1, min(int(search_x_end), int(w)))
        roi = page_bgr[y0:y1, 0:xw]
        if roi.size == 0:
            return (0, 0), []

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        templates_dir = (Path(__file__).resolve().parent / "templates")
        fn4, fn5 = getattr(config, "ZONE1_TEMPLATE_FILES", ("template_4.png", "template_5.png"))
        t4 = cv2.imread(str(templates_dir / fn4), cv2.IMREAD_GRAYSCALE)
        t5 = cv2.imread(str(templates_dir / fn5), cv2.IMREAD_GRAYSCALE)
        if t4 is None or t4.size == 0 or t5 is None or t5.size == 0:
            if not self._logged_zone1_template_fallback:
                log.info(
                    "Zone-1 templates not found/loaded (%s, %s) under %s; falling back to broad above-table zone.",
                    fn4,
                    fn5,
                    str(templates_dir),
                )
                self._logged_zone1_template_fallback = True
            return (0, 0), []

        thr = float(getattr(config, "ZONE1_TEMPLATE_THRESHOLD", 0.75))
        scales = list(getattr(config, "ZONE1_TEMPLATE_SCALES", (0.90, 1.00, 1.10)))

        hit4 = self._best_template_hit(gray, t4, template_file=str(fn4), scales=scales, thr=thr)
        hit5 = self._best_template_hit(gray, t5, template_file=str(fn5), scales=scales, thr=thr)
        if hit4 is None or hit5 is None:
            if not self._logged_zone1_template_fallback:
                log.info(
                    "Zone-1 template matching did not find both anchors (found4=%s found5=%s). Falling back to broad above-table zone.",
                    str(hit4 is not None),
                    str(hit5 is not None),
                )
                self._logged_zone1_template_fallback = True
            return (0, 0), []

        # Compute y-range from anchors (+ padding), convert to page coords.
        # Requirement: template_4 defines the top, template_5 defines the bottom.
        pad = int(getattr(config, "ZONE1_TEMPLATE_PAD_PX", 80))
        t4_y = int(hit4["bbox"][1])
        t5_y2 = int(hit5["bbox"][1] + hit5["bbox"][3])
        if t5_y2 <= t4_y:
            if not self._logged_zone1_template_fallback:
                log.info(
                    "Zone-1 anchors have invalid order (template_4_y=%d, template_5_y2=%d). Falling back to broad above-table zone.",
                    t4_y,
                    t5_y2,
                )
                self._logged_zone1_template_fallback = True
            return (0, 0), []

        yy0 = max(y0, y0 + t4_y - pad)
        yy1 = min(y1, y0 + t5_y2 + pad)
        if yy1 <= yy0:
            return (0, 0), []

        hits = [hit4, hit5]
        return (int(yy0), int(yy1)), hits

    @staticmethod
    def _best_template_hit(
        gray_roi: np.ndarray, tmpl: np.ndarray, *, template_file: str, scales: list[float], thr: float
    ) -> dict | None:
        """
        Return the single best hit dict for a template across multiple scales, or None.
        """
        best = None
        best_score = float(thr)
        for s in scales:
            tw = max(4, int(round(tmpl.shape[1] * float(s))))
            th = max(4, int(round(tmpl.shape[0] * float(s))))
            if tw >= gray_roi.shape[1] or th >= gray_roi.shape[0]:
                continue
            t = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA if float(s) < 1 else cv2.INTER_CUBIC)
            res = cv2.matchTemplate(gray_roi, t, cv2.TM_CCOEFF_NORMED)
            _minv, maxv, _minl, maxl = cv2.minMaxLoc(res)
            if float(maxv) >= best_score:
                x, y = int(maxl[0]), int(maxl[1])
                best_score = float(maxv)
                best = {
                    "bbox": [int(x), int(y), int(tw), int(th)],
                    "confidence": float(maxv),
                    "scale": float(s),
                    "template_file": str(template_file),
                }
        return best

    @staticmethod
    def visualize_zone1_template_hits(
        page_bgr: np.ndarray, hits: list[dict], search_zone: Tuple[int, int], search_x_end: int
    ) -> np.ndarray:
        out = page_bgr.copy()
        y0, y1 = int(search_zone[0]), int(search_zone[1])
        x1 = max(1, min(int(search_x_end), int(out.shape[1])))
        cv2.rectangle(out, (0, y0), (x1 - 1, y1), (255, 0, 0), 2)
        cv2.putText(
            out,
            f"zone1 search: x<={x1}px, y<={y1}px",
            (10, max(30, y0 + 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
            lineType=cv2.LINE_AA,
        )
        for i, h in enumerate(hits):
            x, y, w, hh = h["bbox"]
            # bbox is ROI-relative in our inference; draw at absolute (x, y0+y)
            ax, ay = int(x), int(y0 + int(y))
            cv2.rectangle(out, (ax, ay), (ax + int(w), ay + int(hh)), (255, 0, 255), 3)
            tmpl = str(h.get("template_file", f"t{i+1}"))
            cv2.putText(
                out,
                f"{tmpl} {float(h.get('confidence',0.0)):.2f}",
                (ax, max(0, ay - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2,
                lineType=cv2.LINE_AA,
            )
        if not hits:
            cv2.putText(
                out,
                "ZONE1 TEMPLATE HITS: NONE (fallback zone used)",
                (10, max(70, y0 + 70)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                lineType=cv2.LINE_AA,
            )
        return out

    def _has_table_grid(self, page_bgr: np.ndarray, zone: Tuple[int, int]) -> bool:
        y0, y1 = zone
        roi = page_bgr[y0:y1, :]
        if roi.size == 0:
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 60, 180, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=120, minLineLength=120, maxLineGap=10)
        if lines is None or len(lines) == 0:
            return False

        horiz = 0
        vert = 0
        for x1, y1_, x2, y2_ in lines.reshape(-1, 4):
            dx = abs(int(x2) - int(x1))
            dy = abs(int(y2_) - int(y1_))
            if dx >= 140 and dy <= 6:
                horiz += 1
            elif dy >= 140 and dx <= 6:
                vert += 1

        return (horiz >= 3) and (vert >= 2)


