from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import config
from .ocr_processor import OCRProcessor
from .pdf_loader import PDFCorruptedError, PDFLoader, PDFPasswordError
from .table_detector import TableDetector
from .table_extractor import TableExtractor
from .types import Box, ExtractedRegion, TableStructure
from .utils import Timer, ensure_dir, save_image, save_json, setup_logging
from .validator import Validator
from .zone_detector import ZoneDetector

log = logging.getLogger("extract_handwritten_numbers")


def _get_tqdm():
    """
    Optional progress bar dependency.
    If tqdm isn't installed, return None and we fall back to simple progress prints.
    """
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm
    except Exception:
        return None


def _iter_pdfs(root: Path) -> List[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() == ".pdf" else []
    return [p for p in sorted(root.rglob("*.pdf")) if p.is_file()]


def _out_dir_for_input(input_root: Path, pdf_path: Path, out_root: Path) -> Path:
    """
    Map an input PDF path to an output directory that mirrors the input tree:
      input_root/a/b.pdf -> out_root/a/b/
    """
    if input_root.is_file():
        rel = Path(pdf_path.stem)
    else:
        rel = pdf_path.relative_to(input_root).with_suffix("")
    return out_root / rel


def _draw_zones(page_bgr: np.ndarray, zones: Dict[str, Tuple[int, int]]) -> np.ndarray:
    out = page_bgr.copy()
    h, w = out.shape[:2]
    colors = {"header": (128, 128, 255), "fields": (255, 128, 128), "table": (128, 255, 128)}
    for name, (y0, y1) in zones.items():
        c = colors.get(name, (255, 255, 0))
        cv2.rectangle(out, (0, int(y0)), (w - 1, int(y1)), c, 3)
        cv2.putText(out, name, (10, int(y0) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, c, 3, lineType=cv2.LINE_AA)
    return out


def _draw_zone1_fields_band(page_bgr: np.ndarray, fields_zone: Tuple[int, int]) -> np.ndarray:
    out = page_bgr.copy()
    h, w = out.shape[:2]
    y0, y1 = int(fields_zone[0]), int(fields_zone[1])
    y0 = max(0, min(y0, h))
    y1 = max(0, min(y1, h))
    cv2.rectangle(out, (0, y0), (w - 1, y1), (0, 200, 255), 4)
    cv2.putText(
        out,
        f"ZONE1 fields band: y={y0}..{y1}",
        (10, max(40, y0 + 40)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 200, 255),
        3,
        lineType=cv2.LINE_AA,
    )
    return out


def _draw_logo_overlay(page_bgr: np.ndarray, *, bbox: list | tuple, confidence: float) -> np.ndarray:
    out = page_bgr.copy()
    try:
        x, y, w, h = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    except Exception:
        x, y, w, h = (0, 0, 0, 0)

    # Always print confidence (even when bbox is missing/invalid).
    label = f"LOGO {float(confidence):.2f}"
    cv2.putText(
        out,
        label,
        (10 if (w <= 0 or h <= 0) else x, 40 if (w <= 0 or h <= 0) else max(30, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        3,
        lineType=cv2.LINE_AA,
    )

    if w > 0 and h > 0:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 5)
    return out


def _draw_last_column_overlay(page_bgr: np.ndarray, struct) -> np.ndarray:
    """
    Debug visualization for table extraction:
    - table bbox
    - vertical grid lines
    - highlighted target ("last") column band
    """
    out = page_bgr.copy()
    try:
        bbox = getattr(struct, "bbox", None)
        grid_v = list(getattr(struct, "grid_vertical", []) or [])
        x0, x1 = getattr(struct, "target_column", (0, 0))
        col_idx0 = int(getattr(struct, "target_column_index", -1))
        cols = int(getattr(struct, "cols", 0) or 0)
    except Exception:
        return out

    if bbox is None:
        return out

    try:
        bx, by, bw, bh = int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h)
    except Exception:
        return out

    if bw <= 0 or bh <= 0:
        return out

    # Highlight target column region (semi-transparent)
    x0i, x1i = int(x0), int(x1)
    x0i, x1i = (min(x0i, x1i), max(x0i, x1i))
    if x1i > x0i:
        overlay = out.copy()
        cv2.rectangle(overlay, (x0i, by), (x1i, by + bh), (0, 255, 0), -1)
        out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0.0)

    # Table bbox
    cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (0, 255, 255), 5)

    # Vertical grid lines
    for xv in grid_v:
        xv = int(xv)
        if bx <= xv <= (bx + bw):
            cv2.line(out, (xv, by), (xv, by + bh), (255, 255, 0), 2)

    # Target column outline
    if x1i > x0i:
        cv2.rectangle(out, (x0i, by), (x1i, by + bh), (0, 200, 0), 4)

    cv2.putText(
        out,
        f"LAST COL idx0={col_idx0} cols={cols} x={x0i}..{x1i}",
        (max(10, bx), max(40, by - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 200, 0),
        3,
        lineType=cv2.LINE_AA,
    )
    return out


def structure_results(
    field_regions: List[ExtractedRegion],
    table_regions: List[ExtractedRegion],
    ocr_results: Dict[str, Any],
) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for r in field_regions:
        o = ocr_results.get(r.region_id)
        fields[r.region_id] = {
            "value": getattr(o, "text", "") if o else "",
            "raw_text": getattr(o, "raw_text", "") if o else "",
            "confidence": float(getattr(o, "confidence", 0.0) if o else 0.0),
            "source": (r.meta.get("dot_line") or {}).get("y", None),
            "bbox": [int(r.bbox.x), int(r.bbox.y), int(r.bbox.w), int(r.bbox.h)],
            "meta": (r.meta or {}),
        }

    col3 = []
    for r in table_regions:
        o = ocr_results.get(r.region_id)
        col3.append(
            {
                "id": r.region_id,
                "page": int(r.meta.get("page", 0)),
                "row": int(r.meta.get("row", 0)),
                "column": str(r.meta.get("column_label", "last")),
                "column_index_1based": int(r.meta.get("column_index_1based", 0)),
                "value": getattr(o, "text", "") if o else "",
                "raw_text": getattr(o, "raw_text", "") if o else "",
                "confidence": float(getattr(o, "confidence", 0.0) if o else 0.0),
                "bbox": [int(r.bbox.x), int(r.bbox.y), int(r.bbox.w), int(r.bbox.h)],
                "meta": (r.meta or {}),
            }
        )
    col3.sort(key=lambda x: (int(x["page"]), int(x["row"])))

    return {
        "fields": fields,
        "table": {"total_rows": len(col3), "last_column_values": col3},
    }


def _group_pages_by_document(page_meta: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    docs: Dict[int, List[Dict[str, Any]]] = {}
    for m in (page_meta or []):
        did = int(m.get("document_id", 0) or 0)
        docs.setdefault(did, []).append(m)
    for did in list(docs.keys()):
        docs[did] = sorted(docs[did], key=lambda x: int(x.get("page_num", 0)))
    return docs


def process_form(pdf_path: str, output_dir: str = "output", *, debug: bool = False, ocr_provider: Optional[str] = None) -> Dict[str, Any]:
    """
    Complete pipeline for multi-page form processing.

    Returns a structured JSON-compatible dict (fields + table column 3 + validation + review_queue).
    """
    setup_logging()
    out_root = ensure_dir(output_dir)
    debug_root = ensure_dir(out_root / "debug_output") if debug else None

    timings: Dict[str, float] = {}
    step2_detail: Dict[str, Any] = {}
    manual_review: List[Dict[str, Any]] = []
    # For fallback crops when region templates fail.
    band_pad_px = 20

    def _write_failure_result(err: str) -> Dict[str, Any]:
        """
        Always emit a minimal result.json on failure so the output folder is never empty.
        """
        result = {
            "status": "failed",
            "error": str(err),
            "metadata": {
                "pdf_path": str(pdf_path),
                "pages": None,
                "processing_time": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "timings": {k: round(float(v), 3) for k, v in timings.items()},
                "step2_zone_detection_detail": step2_detail,
                "manual_review": manual_review,
                "extractions": {"fields": 0, "table_column_3": 0, "total": 0},
            },
        }
        try:
            save_json(out_root / "result.json", result)
        except Exception:
            # Best effort; never mask the real failure.
            pass
        return result

    try:
        # STEP 1: Load PDF
        try:
            with Timer("load_pdf") as t:
                pages = PDFLoader(dpi=int(config.PDF_DPI)).load_pdf(pdf_path)
            timings["step1_load_pdf_s"] = float(t.dt or 0.0)
        except PDFPasswordError:
            return _write_failure_result("PDF is password protected")
        except PDFCorruptedError as e:
            return _write_failure_result(f"PDF file is corrupted: {e}")

        # STEP 2: Classify pages and zones (multi-document aware)
        with Timer("zones") as t:
            zone_detector = ZoneDetector()
            try:
                page_meta = zone_detector.classify_pages_with_logo_detection(pages)
            except Exception as e:
                log.warning("Logo-based page classification failed (%s). Falling back to single-document zoning.", str(e))
                page_meta = zone_detector.classify_pages(pages)
                # add minimal multi-doc fields for downstream grouping
                for m in page_meta:
                    m["is_first_page"] = bool(int(m.get("page_num", 0)) == 0)
                    m["document_id"] = 0
                    m["logo_detected"] = False
                    m["logo_confidence"] = 0.0
            step2_detail = getattr(zone_detector, "last_step2_breakdown", {}) or {}
        timings["step2_zone_detection_s"] = float(t.dt or 0.0)

        # Attach a fields-region bbox placeholder for each page.
        # This is the "fields band" bbox (full width) when available; later we may overwrite it
        # with a fallback band crop bbox when region templates fail.
        for m in (page_meta or []):
            try:
                w, h = (m.get("page_size") or (0, 0))
                w = int(w)
                h = int(h)
            except Exception:
                w, h = (0, 0)
            zones = m.get("zones") or {}
            fy0, fy1 = (zones.get("fields") or (0, 0))
            try:
                fy0, fy1 = int(fy0), int(fy1)
            except Exception:
                fy0, fy1 = (0, 0)
            if w > 0 and h > 0 and int(fy1) > int(fy0):
                m["fields_region_bbox"] = [0, int(fy0), int(w), int(fy1 - fy0)]
                m["fields_region_source"] = "region_templates"
            else:
                m["fields_region_bbox"] = [0, 0, 0, 0]
                m["fields_region_source"] = "none"
            # Will be filled during extraction when available.
            m["table_bbox"] = m.get("table_bbox") or [0, 0, 0, 0]
            m["last_column_bbox"] = m.get("last_column_bbox") or [0, 0, 0, 0]
            m["last_column_index_0based"] = int(m.get("last_column_index_0based", -1) or -1)

        # Flag first-page template failures for manual inspection.
        # We intentionally do NOT fall back to other template sets for defining the fields band.
        for m in (page_meta or []):
            is_first = bool(m.get("is_first_page", False)) or int(m.get("page_num", 0) or 0) == 0
            if not bool(is_first):
                continue
            rt = m.get("region_templates") or {}
            rt_status = str(rt.get("status", "") or "")
            fields_zone = (m.get("zones") or {}).get("fields", (0, 0))
            fields_missing = bool(tuple(fields_zone) == (0, 0))
            if fields_missing and rt_status not in ("ok", "n/a"):
                manual_review.append(
                    {
                        "document_id": int(m.get("document_id", 0) or 0),
                        "page": int(m.get("page_num", 0) or 0),
                        "reason": "fields_zone_templates_not_found",
                        "logo_detected": bool(m.get("logo_detected", False)),
                        "logo_confidence": float(m.get("logo_confidence", 0.0) or 0.0),
                        "region_templates": rt,
                    }
                )

        if debug_root is not None:
            # Precompute doc-local page index for filename scheme.
            doc_pages_sorted: dict[int, list[int]] = {}
            for m in page_meta:
                did = int(m.get("document_id", 0) or 0)
                doc_pages_sorted.setdefault(did, []).append(int(m.get("page_num", 0) or 0))
            for did in list(doc_pages_sorted.keys()):
                doc_pages_sorted[did] = sorted(doc_pages_sorted[did])

            for m, img in zip(page_meta, pages):
                p = int(m["page_num"]) + 1
                save_image(debug_root / f"page_{p:02d}_original.jpg", img)

                # Save logo overlay (requested), using multi-doc filename scheme:
                # page_{global}_doc_{doc}_page_{doc_local}
                did0 = int(m.get("document_id", 0) or 0)
                doc_list = doc_pages_sorted.get(did0) or []
                try:
                    doc_local = int(doc_list.index(int(m.get("page_num", 0) or 0)) + 1)
                except Exception:
                    doc_local = 1
                logo_fn = f"page_{p:02d}_doc_{did0+1:02d}_page_{doc_local:02d}_logo.jpg"
                save_image(
                    debug_root / logo_fn,
                    _draw_logo_overlay(
                        img,
                        bbox=m.get("logo_bbox") or (0, 0, 0, 0),
                        confidence=float(m.get("logo_confidence", 0.0) or 0.0),
                    ),
                )
                if bool(m.get("is_first_page", p == 1)):
                    # Always write region template debug on first pages (one per document), even if there are no hits.
                    z1 = m.get("zone1_template_search") or {}
                    search_y = int(z1.get("y_end") or img.shape[0])
                    search_x = int(z1.get("x_end") or img.shape[1])
                    search_y = max(1, min(search_y, int(img.shape[0])))
                    search_x = max(1, min(search_x, int(img.shape[1])))
                    search_zone = (0, int(search_y))
                    save_image(
                        debug_root / f"page_{p:02d}_doc_{did0+1:02d}_page_{doc_local:02d}_region_templates_detected.jpg",
                        _draw_logo_overlay(
                            _draw_zones(
                                zone_detector.visualize_zone1_template_hits(
                                    img, m.get("zone1_template_hits") or [], search_zone, search_x
                                ),
                                m.get("zones") or {},
                            ),
                            bbox=m.get("logo_bbox") or (0, 0, 0, 0),
                            confidence=float(m.get("logo_confidence", 0.0) or 0.0),
                        ),
                    )

            # Dump full page metadata for debugging splits/logo confidence/template status.
            save_json(debug_root / "page_meta.json", page_meta)

        # STEP 3: Extraction (multi-document)
        docs = _group_pages_by_document(page_meta)
        page_meta_by_num: dict[int, dict[str, Any]] = {int(m.get("page_num", 0) or 0): m for m in (page_meta or [])}
        doc_results_meta: List[Dict[str, Any]] = []
        field_regions: List[ExtractedRegion] = []
        table_regions: List[ExtractedRegion] = []
        processed_pages: set[int] = set()
        table_page_times: List[float] = []
        table_pages_attempted: int = 0
        table_pages_failed: int = 0

        with Timer("extract_all_documents") as t_extract:
            for did, doc_pages in sorted(docs.items(), key=lambda kv: int(kv[0])):
                first_pages = [m for m in doc_pages if bool(m.get("is_first_page", False))]
                if not first_pages:
                    # Should not happen (page 0 forced), but be safe.
                    first_pages = [doc_pages[0]]
                first_meta = first_pages[0]
                first_idx = int(first_meta.get("page_num", 0))

                log.info("Processing document %d: pages=%s first_page=%d", int(did), [int(m["page_num"]) + 1 for m in doc_pages], first_idx + 1)

                # TODO
                # If region templates fail on a logo-detected first page:
                # - DO NOT fall back to other template sets for fields zoning.
                # - Skip fields extraction (fields band is undefined), BUT still attempt table extraction.
                rt = first_meta.get("region_templates") or {}
                doc_templates_failed = bool(first_meta.get("logo_detected", False)) and str(rt.get("status", "")) != "ok"
                if doc_templates_failed:
                    msg = (
                        f"Manual review needed: pdf={pdf_path} doc={int(did)} first_page={first_idx+1} "
                        f"region_templates_status={rt.get('status')} used={rt.get('used_templates')} "
                        f"(fields extraction skipped; table extraction will still run)"
                    )
                    log.warning(msg)
                    manual_review.append(
                        {
                            "document_id": int(did),
                            "first_page": int(first_idx),
                            "reason": "region_templates_failed_fields_skipped",
                            "region_templates": rt,
                        }
                    )

                # Fields (only on first page)
                doc_field_regions: List[ExtractedRegion] = []
                prefix = f"doc{int(did)}_p{int(first_idx)+1}_"
                # NOTE: legacy dotted-line/dot detection has been removed from the main pipeline.
                # We OCR a single "fields band" crop per document-first page.
                img0 = pages[first_idx]
                h0, w0 = img0.shape[:2]
                rid = f"{prefix}fields_band"

                # Prefer region-template derived fields band when available.
                zones0 = first_meta.get("zones") or {}
                fy0, fy1 = (zones0.get("fields") or (0, 0))
                try:
                    fy0, fy1 = int(fy0), int(fy1)
                except Exception:
                    fy0, fy1 = (0, 0)

                band_box: Box | None = None
                band_source = "region_templates"
                table_top_y: int | None = None
                header_y1 = int((zones0.get("header") or (0, 0))[1] or 0)
                try:
                    lb = first_meta.get("logo_bbox") or (0, 0, 0, 0)
                    logo_bottom = int(lb[1]) + int(lb[3])
                except Exception:
                    logo_bottom = 0

                if int(fy1) > int(fy0) + 40:
                    band_box = Box(x=0, y=int(fy0), w=int(w0), h=int(fy1 - fy0)).clamp(width=w0, height=h0)
                else:
                    # Fallback: crop from under logo/header down to just above the table.
                    band_source = "fields_band_fallback"
                    y0 = max(int(header_y1), int(logo_bottom)) + int(band_pad_px)
                    td0 = TableDetector()
                    try:
                        tb = td0.detect_table_boundary(img0, (0, int(h0)))
                        table_top_y = int(tb.y)
                        y1 = int(table_top_y) - int(band_pad_px)
                    except Exception as e:
                        manual_review.append(
                            {
                                "document_id": int(did),
                                "first_page": int(first_idx),
                                "reason": "fields_band_table_boundary_failed",
                                "error": str(e),
                            }
                        )
                        y1 = 0
                    if int(y1) > int(y0) + 40:
                        band_box = Box(x=0, y=int(y0), w=int(w0), h=int(y1 - y0)).clamp(width=w0, height=h0)
                    else:
                        manual_review.append(
                            {
                                "document_id": int(did),
                                "first_page": int(first_idx),
                                "reason": "fields_band_ocr_fallback_skipped_invalid_band",
                                "detail": {"y0": int(y0), "y1": int(y1), "header_y1": int(header_y1), "logo_bottom": int(logo_bottom)},
                            }
                        )

                if band_box is not None and int(band_box.w) > 0 and int(band_box.h) > 0:
                    raw_band = img0[band_box.y : band_box.y2, band_box.x : band_box.x2].copy()
                    if raw_band.size:
                        # Light preprocessing: CLAHE + Otsu to help OCR across mixed print/handwriting.
                        gray = cv2.cvtColor(raw_band, cv2.COLOR_BGR2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        gray = clahe.apply(gray)
                        _thr, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        bw = cv2.fastNlMeansDenoising(bw, None, h=10, templateWindowSize=7, searchWindowSize=21)

                        doc_field_regions = [
                            ExtractedRegion(
                                region_id=rid,
                                bbox=band_box,
                                image=bw,
                                raw_image=raw_band,
                                meta={
                                    "kind": "fields_band",
                                    "source": str(band_source),
                                    "y0": int(band_box.y),
                                    "y1": int(band_box.y2),
                                    "table_top_y": (int(table_top_y) if table_top_y is not None else None),
                                    "logo_bbox": list(first_meta.get("logo_bbox") or (0, 0, 0, 0)),
                                    "header_y1": int(header_y1),
                                },
                            )
                        ]
                        field_regions.extend(doc_field_regions)
                        processed_pages.add(int(first_idx))
                        manual_review.append(
                            {
                                "document_id": int(did),
                                "first_page": int(first_idx),
                                "reason": "fields_band_ocr_used",
                                "source": str(band_source),
                                "bbox": [int(band_box.x), int(band_box.y), int(band_box.w), int(band_box.h)],
                            }
                        )
                        pm = page_meta_by_num.get(int(first_idx))
                        if isinstance(pm, dict):
                            pm["fields_region_bbox"] = [int(band_box.x), int(band_box.y), int(band_box.w), int(band_box.h)]
                            pm["fields_region_source"] = str(band_source)

                # Tables (all pages in doc)
                td = TableDetector()
                te = TableExtractor(remove_grid=True)
                row_offset = 0
                doc_table_regions: List[ExtractedRegion] = []

                for m in doc_pages:
                    page_num = int(m.get("page_num", 0))
                    if not bool(m.get("has_table", False)):
                        continue
                    processed_pages.add(int(page_num))
                    img = pages[page_num]
                    zone = m["zones"]["table"]
                    import time

                    t0_page = time.perf_counter()
                    table_pages_attempted += 1
                    try:
                        bbox = td.detect_table_boundary(img, zone)
                        struct = td.parse_table_structure(img, bbox)
                    except Exception as e:
                        table_pages_failed += 1
                        manual_review.append(
                            {
                                "document_id": int(did),
                                "page": int(page_num),
                                "reason": "table_detection_failed",
                                "error": str(e),
                            }
                        )
                        log.warning(
                            "Manual review needed: pdf=%s doc=%d page=%d table detection failed (%s). Skipping table extraction for this page.",
                            str(pdf_path),
                            int(did),
                            int(page_num) + 1,
                            str(e),
                        )
                        continue
                    finally:
                        table_page_times.append(float(time.perf_counter() - t0_page))

                    # Persist per-page table/last-column bboxes into page metadata (even when not debugging).
                    pm = page_meta_by_num.get(int(page_num))
                    if isinstance(pm, dict):
                        try:
                            tb = getattr(struct, "bbox", None)
                            if tb is not None:
                                pm["table_bbox"] = [int(tb.x), int(tb.y), int(tb.w), int(tb.h)]
                            x0, x1 = getattr(struct, "target_column", (0, 0))
                            x0i, x1i = int(x0), int(x1)
                            x0i, x1i = (min(x0i, x1i), max(x0i, x1i))
                            if tb is not None and int(tb.h) > 0 and int(x1i) > int(x0i):
                                pm["last_column_bbox"] = [int(x0i), int(tb.y), int(x1i - x0i), int(tb.h)]
                            pm["last_column_index_0based"] = int(getattr(struct, "target_column_index", -1) or -1)
                        except Exception:
                            # Best effort; never break extraction for metadata.
                            pass

                    if debug_root is not None:
                        # Save per-page last-column overlay (uses per-page target column).
                        p = int(page_num) + 1
                        try:
                            doc_local = int([int(mm.get("page_num", 0)) for mm in doc_pages].index(int(page_num)) + 1)
                        except Exception:
                            doc_local = 1
                        fn = f"page_{p:02d}_doc_{int(did)+1:02d}_page_{doc_local:02d}_last_column.jpg"
                        save_image(debug_root / fn, _draw_last_column_overlay(img, struct))

                        # Also add last-column overlay to the region-template debug image (first pages only).
                        if bool(m.get("is_first_page", False)):
                            z1 = m.get("zone1_template_search") or {}
                            search_y = int(z1.get("y_end") or img.shape[0])
                            search_x = int(z1.get("x_end") or img.shape[1])
                            search_y = max(1, min(search_y, int(img.shape[0])))
                            search_x = max(1, min(search_x, int(img.shape[1])))
                            search_zone = (0, int(search_y))

                            composed = _draw_last_column_overlay(img, struct)
                            composed = zone_detector.visualize_zone1_template_hits(
                                composed, m.get("zone1_template_hits") or [], search_zone, search_x
                            )
                            composed = _draw_zones(composed, m.get("zones") or {})
                            composed = _draw_logo_overlay(
                                composed,
                                bbox=m.get("logo_bbox") or (0, 0, 0, 0),
                                confidence=float(m.get("logo_confidence", 0.0) or 0.0),
                            )
                            rt_fn = f"page_{p:02d}_doc_{int(did)+1:02d}_page_{doc_local:02d}_region_templates_detected.jpg"
                            save_image(debug_root / rt_fn, composed)

                    page_cells = te.extract_target_column_cells(img, struct, page_num, row_offset=row_offset, skip_header_rows=1)
                    doc_table_regions.extend(page_cells)
                    row_offset += len(page_cells)

                table_regions.extend(doc_table_regions)

                doc_results_meta.append(
                    {
                        "document_id": int(did),
                        "pages": [int(m["page_num"]) for m in doc_pages],
                        "first_page": int(first_idx),
                        "logo_detected": bool(first_meta.get("logo_detected", False)),
                        "logo_confidence": float(first_meta.get("logo_confidence", 0.0) or 0.0),
                        "needs_manual_review": bool(doc_templates_failed),
                        "status": "ok",
                        "warnings": (["region_templates_failed_fields_skipped"] if bool(doc_templates_failed) else []),
                        "extractions": {"fields": int(len(doc_field_regions)), "table_column_3": int(len(doc_table_regions))},
                    }
                )

        timings["step3_extract_all_documents_s"] = float(t_extract.dt or 0.0)

        # If debugging, overwrite page_meta.json with extraction-enriched bboxes.
        if debug_root is not None:
            try:
                save_json(debug_root / "page_meta.json", page_meta)
            except Exception:
                pass

        # STEP 4: Batch OCR
        with Timer("ocr") as t:
            all_regions = field_regions + table_regions
            images = [r.image for r in all_regions]
            ids = [r.region_id for r in all_regions]
            ocr = OCRProcessor(provider=ocr_provider or config.OCR_PROVIDER, languages=config.OCR_LANGUAGES)
            ocr_results = ocr.batch_ocr(images, ids)
        timings["step4_batch_ocr_s"] = float(t.dt or 0.0)

        # STEP 5: Structure + validate (per-document)
        with Timer("validate") as t:
            docs = _group_pages_by_document(page_meta)
            validator = Validator()
            documents_out: List[Dict[str, Any]] = []
            doc_status_by_id: Dict[int, str] = {int(m.get("document_id", 0)): str(m.get("status", "ok")) for m in (doc_results_meta or [])}
            doc_needs_review_by_id: Dict[int, bool] = {int(m.get("document_id", 0)): bool(m.get("needs_manual_review", False)) for m in (doc_results_meta or [])}

            # Build fast lookups by page for region assignment
            fields_by_doc: Dict[int, List[ExtractedRegion]] = {}
            for r in field_regions:
                # region_id prefix encodes doc id as "doc{did}_..."
                did = 0
                if r.region_id.startswith("doc"):
                    try:
                        did = int(r.region_id.split("_", 1)[0].replace("doc", ""))
                    except Exception:
                        did = 0
                fields_by_doc.setdefault(did, []).append(r)

            tables_by_doc: Dict[int, List[ExtractedRegion]] = {}
            for did, doc_pages in docs.items():
                page_nums = {int(m["page_num"]) for m in doc_pages}
                tables_by_doc[int(did)] = [r for r in table_regions if int((r.meta.get("page", 0) or 0)) - 1 in page_nums]

            for did, doc_pages in sorted(docs.items(), key=lambda kv: int(kv[0])):
                first_pages = [m for m in doc_pages if bool(m.get("is_first_page", False))]
                first_idx = int((first_pages[0] if first_pages else doc_pages[0]).get("page_num", 0))
                f_regs = fields_by_doc.get(int(did), [])
                t_regs = tables_by_doc.get(int(did), [])

                if str(doc_status_by_id.get(int(did), "ok")) != "ok":
                    # Skip validation/structuring for failed docs; keep a clear marker in output.
                    documents_out.append(
                        {
                            "document_id": int(did),
                            "pages": [int(m["page_num"]) for m in doc_pages],
                            "first_page": int(first_idx),
                            "status": "failed",
                            "error": "region_templates_failed_skip_document",
                        }
                    )
                    continue

                structured_doc = structure_results(f_regs, t_regs, ocr_results)
                validated_doc = validator.validate_results(structured_doc)
                documents_out.append(
                    {
                        "document_id": int(did),
                        "pages": [int(m["page_num"]) for m in doc_pages],
                        "first_page": int(first_idx),
                        "needs_manual_review": bool(doc_needs_review_by_id.get(int(did), False)),
                        **validated_doc,
                    }
                )

        timings["step5_validation_s"] = float(t.dt or 0.0)

        total_s = float(sum(timings.values()))
        # Summary stats
        total_pages = int(len(pages))
        total_docs = int(len(doc_results_meta) or 1)
        avg_s_per_page = float(total_s / max(1, total_pages))
        timings["avg_total_s_per_page"] = float(avg_s_per_page)
        timings["avg_step2_zone_detection_s_per_page"] = float(timings.get("step2_zone_detection_s", 0.0) / max(1, total_pages))
        timings["avg_step3_extraction_s_per_page"] = float(timings.get("step3_extract_all_documents_s", 0.0) / max(1, total_pages))
        if table_page_times:
            timings["avg_table_detection_s_per_table_page"] = float(sum(table_page_times) / max(1, len(table_page_times)))
        log.info("Docs found: %d", total_docs)
        log.info("Pages: %d  processed_pages: %d", total_pages, int(len(processed_pages)))
        log.info("Avg total time: %.3fs/page", avg_s_per_page)
        if table_page_times:
            log.info(
                "Table detection: attempted=%d failed=%d avg=%.3fs/table-page",
                int(table_pages_attempted),
                int(table_pages_failed),
                float(sum(table_page_times) / max(1, len(table_page_times))),
            )
        result: Dict[str, Any] = {
            "metadata": {
                "pdf_path": str(pdf_path),
                "pages": int(len(pages)),
                "pages_processed": int(len(processed_pages)),
                "total_documents": int(len(doc_results_meta) or 1),
                "processing_time": round(total_s, 3),
                "avg_total_s_per_page": float(avg_s_per_page),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "timings": {k: round(float(v), 3) for k, v in timings.items()},
                "step2_zone_detection_detail": step2_detail,
                "manual_review": manual_review,
                "page_layout": [
                    {
                        "page_num": int(m.get("page_num", 0) or 0),
                        "document_id": int(m.get("document_id", 0) or 0),
                        "is_first_page": bool(m.get("is_first_page", False)),
                        "fields_region_bbox": list(m.get("fields_region_bbox") or [0, 0, 0, 0]),
                        "fields_region_source": str(m.get("fields_region_source", "none") or "none"),
                        "table_bbox": list(m.get("table_bbox") or [0, 0, 0, 0]),
                        "last_column_bbox": list(m.get("last_column_bbox") or [0, 0, 0, 0]),
                        "last_column_index_0based": int(m.get("last_column_index_0based", -1) or -1),
                    }
                    for m in (page_meta or [])
                ],
                "extractions": {
                    "fields": len(field_regions),
                    "table_column_3": len(table_regions),
                    "total": len(field_regions) + len(table_regions),
                },
            },
            "documents": documents_out,
            "documents_meta": doc_results_meta,
        }

        # Backwards compatibility: if we only detected one document, keep legacy top-level keys.
        if len(documents_out) == 1:
            for k in ("fields", "table", "validation", "review_queue"):
                if k in documents_out[0]:
                    result[k] = documents_out[0][k]

        save_json(out_root / "result.json", result)
        if debug_root is not None:
            save_json(debug_root / "timing_breakdown.json", timings)
            save_json(debug_root / "ocr_results.json", {k: v.__dict__ for k, v in ocr_results.items()})

        log.info("Step 1: Loaded %d pages (%.3fs)", len(pages), timings.get("step1_load_pdf_s", 0.0))
        log.info(
            "Step 3: Extracted %d fields + %d table cells across %d document(s) (%.3fs)",
            len(field_regions),
            len(table_regions),
            int(len(doc_results_meta) or 1),
            timings.get("step3_extract_all_documents_s", 0.0),
        )
        log.info("Total processing time: %.3fs", total_s)
        return result
    except Exception as e:
        # Best-effort: write a failure result.json, then re-raise so batch mode records status=error.
        _write_failure_result(str(e))
        raise


def process_inputs(
    input_path: str,
    out_root: str = "output",
    *,
    debug: bool = False,
    progress: bool = True,
    force: bool = False,
    max_files: int = 0,
    ocr_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Walk a directory (or process a single file) and run `process_form()` for each PDF.

    Output mirrors the input structure under `out_root`:
      input_root/a/b.pdf -> out_root/a/b/result.json
    """
    setup_logging()
    in_root = Path(input_path)
    out_root_p = ensure_dir(out_root)

    pdfs = _iter_pdfs(in_root)
    if not pdfs:
        raise SystemExit(f"No PDFs found under: {in_root}")

    if max_files and max_files > 0:
        pdfs = pdfs[: int(max_files)]

    tqdm_mod = _get_tqdm() if progress else None
    it = tqdm_mod(pdfs, desc="Processing PDFs", unit="pdf") if tqdm_mod else pdfs

    processed = 0
    skipped = 0
    failed = 0
    rows: list[dict[str, Any]] = []

    for p in it:
        out_dir = _out_dir_for_input(in_root, p, out_root_p)
        result_path = out_dir / "result.json"
        if result_path.exists() and not force:
            skipped += 1
            continue

        try:
            r = process_form(str(p), str(out_dir), debug=bool(debug), ocr_provider=ocr_provider)
            # process_form can return a structured failure dict (status="failed") without throwing.
            status = str(r.get("status") or "ok")
            row = {
                "pdf": str(p),
                "out_dir": str(out_dir),
                "status": status,
                "pages": r.get("metadata", {}).get("pages", None) if isinstance(r, dict) else None,
                "processing_time": r.get("metadata", {}).get("processing_time", None) if isinstance(r, dict) else None,
            }
            if status != "ok":
                row["error"] = r.get("error", None) if isinstance(r, dict) else None
                rows.append(row)
                failed += 1
            else:
                rows.append(row)
                processed += 1
        except Exception as e:
            failed += 1
            rows.append({"pdf": str(p), "out_dir": str(out_dir), "status": "error", "error": str(e)})

        if tqdm_mod is None and progress:
            # simple fallback status line
            done = processed + skipped + failed
            if done % 5 == 0 or done == len(pdfs):
                log.info("Progress %d/%d (ok=%d skipped=%d failed=%d)", done, len(pdfs), processed, skipped, failed)

    summary = {
        "input": str(in_root),
        "out_root": str(out_root_p),
        "total": int(len(pdfs)),
        "ok": int(processed),
        "skipped": int(skipped),
        "failed": int(failed),
        "items": rows,
    }
    save_json(out_root_p / "batch_summary.json", summary)
    return summary


def _cli(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Multi-page Thai form OCR (dotted lines + table column 3 only).")
    ap.add_argument("--input", default=None, help="Input PDF path OR a directory to scan recursively for PDFs.")
    ap.add_argument("--pdf", default=None, help="Alias for --input (backwards compatible).")
    ap.add_argument("--out", default="output", help="Output root directory. Default: output/")
    ap.add_argument("--debug", action="store_true", help="Save intermediate debug images.")
    ap.add_argument("--ocr-provider", default=None, choices=["google", "tesseract"],
                     help="OCR provider override. Default: config value (google).")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bar / progress prints.")
    ap.add_argument("--force", action="store_true", help="Reprocess even if output result.json already exists.")
    ap.add_argument("--max-files", type=int, default=0, help="0 = no limit. Otherwise process only first N PDFs found.")
    args = ap.parse_args(argv)

    in_path = args.input or args.pdf
    if not in_path:
        raise SystemExit("Missing required flag: --input (or legacy alias: --pdf)")

    _ = process_inputs(
        str(in_path),
        str(args.out),
        debug=bool(args.debug),
        progress=(not bool(args.no_progress)),
        force=bool(args.force),
        max_files=int(args.max_files),
        ocr_provider=args.ocr_provider,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


