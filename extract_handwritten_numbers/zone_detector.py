from __future__ import annotations

import logging
from typing import Dict, List, Tuple
from pathlib import Path

import cv2
import numpy as np

from . import config
from .logo_detector import LogoDetector

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
        # Populated after classify_* calls; used by main.py to include step-2 breakdowns in result.json.
        self.last_step2_breakdown: dict = {}
        # Cache loaded template images to avoid re-reading from disk on every call.
        self._template_cache: Dict[str, np.ndarray | None] = {}

    def _load_template(self, filename: str) -> np.ndarray | None:
        """Load a grayscale template image, caching the result."""
        if filename in self._template_cache:
            return self._template_cache[filename]
        templates_dir = Path(__file__).resolve().parent / "templates"
        t = cv2.imread(str(templates_dir / filename), cv2.IMREAD_GRAYSCALE)
        if t is not None and t.size == 0:
            t = None
        self._template_cache[filename] = t
        return t

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
        """
        Backwards-compatible single-document classification.
        For multi-document PDFs, use `classify_pages_with_logo_detection()`.
        """
        import time

        t_total0 = time.perf_counter()
        out: List[Dict] = []
        t_region_templates = 0.0
        t_has_table = 0.0
        n_region_pages = 0
        n_has_table_pages = 0
        for i, page in enumerate(pages):
            h, w = page.shape[:2]
            if i == 0:
                # Page 1:
                # - No header zone (template/bullet approach makes header zoning unnecessary)
                # - Fields zone inferred from region templates (region_begin/end) when available
                header = (0, 0)
                table = self._zone_px(h, config.TABLE_ZONE)
                # Region templates search window:
                y_end = int(round(float(h) * float(getattr(config, "REGION_TEMPLATE_SEARCH_Y_FRAC", 0.65))))
                y_end = max(1, min(y_end, h))
                x_end = int(round(float(w) * float(getattr(config, "REGION_TEMPLATE_SEARCH_X_FRAC", 1.0))))
                x_end = max(1, min(x_end, w))
                t0 = time.perf_counter()
                fields, zone1_hits, region_status = self._infer_fields_zone_from_region_templates(
                    page, search_zone=(0, int(y_end)), search_x_end=int(x_end)
                )
                t_region_templates += float(time.perf_counter() - t0)
                n_region_pages += 1
            else:
                # Continuation pages: treat as table-only (full page).
                header = (0, 0)
                fields = (0, 0)
                table = (0, h)
                zone1_hits = []
                x_end = 0
                y_end = 0
                region_status = {"status": "n/a"}

            t0 = time.perf_counter()
            has_table = self._has_table_grid(page, table)
            t_has_table += float(time.perf_counter() - t0)
            n_has_table_pages += 1
            has_fields = bool(i == 0 and fields != (0, 0))

            out.append(
                {
                    "page_num": i,
                    "zones": {"header": header, "fields": fields, "table": table},
                    "has_fields": bool(has_fields),
                    "has_table": bool(has_table),
                    "page_size": (w, h),
                    "zone1_template_hits": zone1_hits,
                    "zone1_template_search": {"x_end": int(x_end), "y_end": int(y_end)},
                    "region_templates": region_status,
                }
            )
        t_total = float(time.perf_counter() - t_total0)
        self.last_step2_breakdown = {
            "mode": "single_document",
            "total_s": float(t_total),
            "region_templates_s": float(t_region_templates),
            "region_templates_pages": int(n_region_pages),
            "region_templates_avg_s_per_page": float(t_region_templates / max(1, n_region_pages)),
            "has_table_grid_s": float(t_has_table),
            "has_table_grid_pages": int(n_has_table_pages),
            "has_table_grid_avg_s_per_page": float(t_has_table / max(1, n_has_table_pages)),
        }
        return out

    def classify_pages_with_logo_detection(self, pages: List[np.ndarray]) -> List[Dict]:
        """
        Multi-document classification:
        - Detect logo in top band to identify first pages (starts of new forms).
        - Assign document_id by splitting at detected first pages.
        """
        if not pages:
            return []

        min_pages_for_logo = int(getattr(config, "MIN_PAGES_FOR_LOGO_DETECTION", 3))
        use_logo = len(pages) >= int(min_pages_for_logo)
        if not bool(use_logo):
            log.info(
                "Skipping logo detection: pages=%d < MIN_PAGES_FOR_LOGO_DETECTION=%d (treating as single document).",
                int(len(pages)),
                int(min_pages_for_logo),
            )
        logo = LogoDetector()

        min_gap = int(getattr(config, "LOGO_MIN_PAGE_GAP", 1))
        always_page0 = bool(getattr(config, "ALWAYS_TREAT_PAGE0_AS_FIRST", True))

        doc_id = 0
        last_first_page = -10_000
        out: List[Dict] = []

        import time

        t_total0 = time.perf_counter()
        logo_time_total = 0.0
        region_time_total = 0.0
        has_table_time_total = 0.0
        n_logo_pages = 0
        n_region_pages = 0
        n_has_table_pages = 0
        skip_logo_detection = False
        for i, page in enumerate(pages):
            h, w = page.shape[:2]

            hit = {"has_logo": False, "confidence": 0.0, "position": (0, 0), "bbox": (0, 0, 0, 0), "scale": 1.0}
            if use_logo:
                try:
                    if not bool(skip_logo_detection):
                        t0 = time.perf_counter()
                        hit = logo.detect_logo(page)
                        logo_time_total += float(time.perf_counter() - t0)
                        n_logo_pages += 1
                    else:
                        # If previous page was marked first/logo, the immediate next page is treated as continuation.
                        hit = {"has_logo": False, "confidence": 0.0, "position": (0, 0), "bbox": (0, 0, 0, 0), "scale": 1.0}
                except Exception as e:
                    log.warning("Logo detection failed on page %d (%s). Continuing without logo for this page.", i + 1, str(e))
                    hit = {"has_logo": False, "confidence": 0.0, "position": (0, 0), "bbox": (0, 0, 0, 0), "scale": 1.0}

            is_first_page = bool(hit.get("has_logo", False))
            if i == 0 and always_page0:
                if not is_first_page and use_logo:
                    log.warning("Logo not detected on page 1, but treating page 1 as first page (ALWAYS_TREAT_PAGE0_AS_FIRST).")
                is_first_page = True

            # Debounce: avoid splitting if logos are detected too close together (likely false positives).
            if i > 0 and is_first_page and (i - last_first_page) <= int(min_gap):
                log.warning(
                    "Logo detected too close to previous first page (page %d and %d). Treating page %d as continuation (likely false positive).",
                    last_first_page + 1,
                    i + 1,
                    i + 1,
                )
                is_first_page = False

            if i > 0 and is_first_page:
                doc_id += 1
                last_first_page = i
            elif i == 0 and is_first_page:
                last_first_page = i

            # If this page is marked as first page (forced or logo-detected), skip detection on the immediate next page.
            skip_logo_detection = bool(is_first_page)

            if is_first_page:
                header_y1 = int(round(float(h) * float(getattr(config, "LOGO_SEARCH_Y_FRAC", 0.20))))
                header_y1 = max(1, min(header_y1, h))
                header = (0, int(header_y1))
                # Table search zone:
                # - If logo was actually detected on this page, restrict to lower 50% (avoid header artifacts).
                # - If this is a forced first page without logo, search whole page (more robust).
                if bool(hit.get("has_logo", False)):
                    table = (int(round(0.50 * float(h))), int(h))
                else:
                    table = (0, int(h))
                y_end = int(
                    round(float(h) * float(getattr(config, "REGION_TEMPLATE_SEARCH_Y_FRAC", getattr(config, "ZONE1_TEMPLATE_SEARCH_Y_FRAC", 0.65))))
                )
                y_end = max(1, min(y_end, h))
                x_end = int(
                    round(float(w) * float(getattr(config, "REGION_TEMPLATE_SEARCH_X_FRAC", getattr(config, "ZONE1_TEMPLATE_SEARCH_X_FRAC", 0.5))))
                )
                x_end = max(1, min(x_end, w))
                region_status: dict = {"status": "skipped"}
                # Use region templates to define the fields band on FIRST pages.
                # If anchors are not found, do NOT fall back to a broad band; dotted-line detection must be skipped.
                t0 = time.perf_counter()
                fields, zone1_hits, region_status = self._infer_fields_zone_from_region_templates(
                    page, search_zone=(0, int(y_end)), search_x_end=int(x_end)
                )
                region_time_total += float(time.perf_counter() - t0)
                n_region_pages += 1
            else:
                header = (0, 0)
                fields = (0, 0)
                # Table search zone:
                # - If logo is detected on this page, restrict to lower 50% (avoid header artifacts).
                # - Otherwise, search whole page.
                if bool(hit.get("has_logo", False)):
                    table = (int(round(0.50 * float(h))), int(h))
                else:
                    table = (0, int(h))
                zone1_hits = []
                x_end = 0
                y_end = 0
                region_status = {"status": "n/a"}

            t0 = time.perf_counter()
            has_table = self._has_table_grid(page, table)
            has_table_time_total += float(time.perf_counter() - t0)
            n_has_table_pages += 1
            has_fields = bool(is_first_page and fields != (0, 0))

            out.append(
                {
                    "page_num": i,
                    "is_first_page": bool(is_first_page),
                    "document_id": int(doc_id),
                    "logo_detected": bool(hit.get("has_logo", False)),
                    "logo_confidence": float(hit.get("confidence", 0.0) or 0.0),
                    "logo_bbox": list(hit.get("bbox", (0, 0, 0, 0))),
                    "zones": {"header": header, "fields": fields, "table": table},
                    "has_fields": bool(has_fields),
                    "has_table": bool(has_table),
                    "page_size": (w, h),
                    "zone1_template_hits": zone1_hits,
                    "zone1_template_search": {"x_end": int(x_end), "y_end": int(y_end)},
                    "region_templates": region_status,
                }
            )

        # Sanity warnings for false positives / misconfigurations.
        first_pages = [m for m in out if m.get("is_first_page")]
        if use_logo and len(pages) >= 3:
            if len(first_pages) == 1 and not bool(first_pages[0].get("logo_detected", False)) and len(pages) >= 3:
                log.warning("No logos detected in multi-page PDF; treating as single document (page 1 forced as first).")
            if len(first_pages) > int(len(pages) * 0.5):
                log.warning(
                    "Suspicious: %d/%d pages marked as first pages (>50%%). Logo template may be too generic; consider increasing LOGO_DETECTION_THRESHOLD.",
                    len(first_pages),
                    len(pages),
                )

        if use_logo:
            # helpful for performance debugging
            log.info(
                "Logo detection time total: %.3fs for %d pages (avg %.3fs/page)",
                logo_time_total,
                int(n_logo_pages),
                logo_time_total / max(1, n_logo_pages),
            )

        t_total = float(time.perf_counter() - t_total0)
        self.last_step2_breakdown = {
            "mode": "multi_document_logo",
            "logo_detection_enabled": bool(use_logo),
            "min_pages_for_logo_detection": int(min_pages_for_logo),
            "total_s": float(t_total),
            "logo_detection_s": float(logo_time_total),
            "logo_detection_pages": int(n_logo_pages),
            "logo_detection_avg_s_per_page": float(logo_time_total / max(1, n_logo_pages)),
            "region_templates_s": float(region_time_total),
            "region_templates_pages": int(n_region_pages),
            "region_templates_avg_s_per_page": float(region_time_total / max(1, n_region_pages)),
            "has_table_grid_s": float(has_table_time_total),
            "has_table_grid_pages": int(n_has_table_pages),
            "has_table_grid_avg_s_per_page": float(has_table_time_total / max(1, n_has_table_pages)),
        }

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

        fn4, fn5 = getattr(config, "ZONE1_TEMPLATE_FILES", ("template_4.png", "template_5.png"))
        t4 = self._load_template(fn4)
        t5 = self._load_template(fn5)
        if t4 is None or t5 is None:
            if not self._logged_zone1_template_fallback:
                log.info(
                    "Zone-1 templates not found/loaded (%s, %s) under %s; falling back to broad above-table zone.",
                    fn4,
                    fn5,
                    str(templates_dir),
                )
                self._logged_zone1_template_fallback = True
            return (0, 0), []

        thr = float(getattr(config, "ZONE1_TEMPLATE_THRESHOLD", 0.6))
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

    def _infer_fields_zone_from_region_templates(
        self, page_bgr: np.ndarray, *, search_zone: Tuple[int, int], search_x_end: int
    ) -> tuple[Tuple[int, int], list[dict], dict]:
        """
        For logo-detected first pages: infer fields y-range using region begin/end anchors.

        Try in order:
        - set1: REGION_TEMPLATE_FILES_1 = (region_begin, region_end)
        - if rejected/not found: set2: REGION_TEMPLATE_FILES_2 = (region_begin_2, region_end_2)

        Returns:
        - (y0,y1) absolute pixels or (0,0) when unavailable/not matched
        - hits list (ROI-relative bboxes, same format as zone-1 hits) for debug overlays
        - status dict that includes which filenames were tried and why
        """
        y0, y1 = int(search_zone[0]), int(search_zone[1])
        h, w = page_bgr.shape[:2]
        y0 = max(0, min(y0, h))
        y1 = max(0, min(y1, h))
        if y1 <= y0:
            return (0, 0), [], {"status": "invalid_search_zone"}

        xw = max(1, min(int(search_x_end), int(w)))
        roi = page_bgr[y0:y1, 0:xw]
        if roi.size == 0:
            return (0, 0), [], {"status": "empty_roi"}

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        thr = float(getattr(config, "REGION_TEMPLATE_THRESHOLD", 0.70))
        # Prefer range sweep to match debug script behavior.
        scales: list[float] = []
        rng = getattr(config, "REGION_TEMPLATE_SCALE_RANGE", None)
        if isinstance(rng, (tuple, list)) and len(rng) == 3:
            try:
                a, b, step = float(rng[0]), float(rng[1]), float(rng[2])
                if step > 0 and a > 0 and b >= a:
                    x = a
                    for _ in range(2000):
                        if x > b + 1e-9:
                            break
                        scales.append(float(x))
                        x += step
            except Exception:
                scales = []
        if not scales:
            scales = list(getattr(config, "REGION_TEMPLATE_SCALES", (0.80, 0.90, 1.00, 1.10, 1.20)))
        pad = int(getattr(config, "REGION_TEMPLATE_PAD_PX", 60))

        files1 = tuple(getattr(config, "REGION_TEMPLATE_FILES_1", ("region_begin.png", "region_end.png")))
        files2 = tuple(getattr(config, "REGION_TEMPLATE_FILES_2", ("region_begin_2.png", "region_end_2.png")))

        tried: list[dict] = []

        def _load_gray(fn: str) -> np.ndarray | None:
            return self._load_template(fn)

        def _try_pair(pair: tuple[str, str], *, tag: str) -> tuple[Tuple[int, int], list[dict]] | None:
            fn_begin, fn_end = str(pair[0]), str(pair[1])
            t_begin = _load_gray(fn_begin)
            t_end = _load_gray(fn_end)
            entry: dict = {
                "tag": str(tag),
                "files": [fn_begin, fn_end],
                "loaded": [t_begin is not None, t_end is not None],
            }
            tried.append(entry)
            if t_begin is None or t_end is None:
                entry["reason"] = "template_missing"
                return None

            # Always compute best score for diagnostics, even if below threshold.
            best_b = self._best_template_hit_any(gray, t_begin, template_file=str(fn_begin), scales=scales)
            best_e = self._best_template_hit_any(gray, t_end, template_file=str(fn_end), scales=scales)
            entry["best"] = [best_b, best_e]
            acc_b = bool(best_b is not None and float(best_b.get("confidence", 0.0)) >= float(thr))
            acc_e = bool(best_e is not None and float(best_e.get("confidence", 0.0)) >= float(thr))
            entry["accepted"] = [bool(acc_b), bool(acc_e)]
            entry["matched"] = [bool(acc_b), bool(acc_e)]
            if not (acc_b and acc_e):
                entry["reason"] = "below_threshold_or_not_found"
                return None

            # Use the accepted "best" hits.
            hit_b = best_b
            hit_e = best_e

            y_begin = int(hit_b["bbox"][1])
            y_end2 = int(hit_e["bbox"][1] + hit_e["bbox"][3])
            if y_end2 <= y_begin:
                entry["reason"] = "invalid_order"
                return None

            yy0 = max(y0, y0 + y_begin - pad)
            yy1 = min(y1, y0 + y_end2 + pad)
            if yy1 <= yy0:
                entry["reason"] = "empty_band"
                return None

            entry["confidence"] = [float(hit_b.get("confidence", 0.0)), float(hit_e.get("confidence", 0.0))]
            return (int(yy0), int(yy1)), [hit_b, hit_e]

        used: str | None = None
        got = _try_pair((str(files1[0]), str(files1[1])), tag="set1")
        if got is not None:
            used = "set1"
        else:
            got = _try_pair((str(files2[0]), str(files2[1])), tag="set2")
            if got is not None:
                used = "set2"

        if got is None:
            return (
                (0, 0),
                [],
                {
                    "status": "none",
                    "method": "region_templates",
                    "used": None,
                    "used_templates": [],
                    "threshold": float(thr),
                    "scales": list(scales),
                    "tried": tried,
                },
            )

        band, hits = got
        if used == "set2":
            used_files = [str(files2[0]), str(files2[1])]
        else:
            used_files = [str(files1[0]), str(files1[1])]
        return (
            band,
            hits,
            {
                "status": "ok",
                "method": "region_templates",
                "used": used,
                "used_templates": list(used_files),
                "threshold": float(thr),
                "scales": list(scales),
                "tried": tried,
            },
        )

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
    def _best_template_hit_any(gray_roi: np.ndarray, tmpl: np.ndarray, *, template_file: str, scales: list[float]) -> dict | None:
        """
        Return the single best hit dict for a template across multiple scales, regardless of threshold.
        Returns None only if the template never fit inside the ROI at any scale.
        """
        best = None
        best_score = float("-inf")
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
                "TEMPLATE HITS: NONE (manual review flagged in result.json)",
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

        return (horiz >= 3) and (vert >= 3)


