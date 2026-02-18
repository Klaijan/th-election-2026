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
from .field_extractor import FieldExtractor
from .ocr_processor import OCRProcessor
from .pdf_loader import PDFCorruptedError, PDFLoader, PDFPasswordError
from .table_detector import TableDetector
from .table_extractor import TableExtractor
from .types import Box, ExtractedRegion, TableStructure
from .utils import Timer, ensure_dir, save_image, save_json, setup_logging
from .validator import Validator
from .zone_detector import ZoneDetector
from .dot_detector import DotDetector

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
        return out
    if w <= 0 or h <= 0:
        return out
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 5)
    cv2.putText(
        out,
        f"LOGO {float(confidence):.2f}",
        (x, max(30, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
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
            "confidence": float(getattr(o, "confidence", 0.0) if o else 0.0),
            "source": (r.meta.get("dot_line") or {}).get("y", None),
            "bbox": [int(r.bbox.x), int(r.bbox.y), int(r.bbox.w), int(r.bbox.h)],
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
                "confidence": float(getattr(o, "confidence", 0.0) if o else 0.0),
                "bbox": [int(r.bbox.x), int(r.bbox.y), int(r.bbox.w), int(r.bbox.h)],
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


def process_form(pdf_path: str, output_dir: str = "output", *, debug: bool = False) -> Dict[str, Any]:
    """
    Complete pipeline for multi-page form processing.

    Returns a structured JSON-compatible dict (fields + table column 3 + validation + review_queue).
    """
    setup_logging()
    out_root = ensure_dir(output_dir)
    debug_root = ensure_dir(out_root / "debug_output") if debug else None

    timings: Dict[str, float] = {}
    step2_detail: Dict[str, Any] = {}

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
                save_image(debug_root / f"page_{p:02d}_zones_marked.jpg", _draw_zones(img, m["zones"]))

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
                    # Always write zone-1 template debug on first pages (one per document), even if there are no hits.
                    z1 = m.get("zone1_template_search") or {}
                    search_y = int(z1.get("y_end") or img.shape[0])
                    search_x = int(z1.get("x_end") or img.shape[1])
                    search_y = max(1, min(search_y, int(img.shape[0])))
                    search_x = max(1, min(search_x, int(img.shape[1])))
                    search_zone = (0, int(search_y))
                    save_image(
                        debug_root / f"page_{p:02d}_zone1_templates_detected.jpg",
                        zone_detector.visualize_zone1_template_hits(
                            img, m.get("zone1_template_hits") or [], search_zone, search_x
                        ),
                    )
                    # Also save with doc-style naming (helps multi-form PDFs). Note: zone1_template_hits will contain
                    # region_begin/end hits on logo-detected pages (by design).
                    save_image(
                        debug_root / f"page_{p:02d}_doc_{did0+1:02d}_page_{doc_local:02d}_region_templates_detected.jpg",
                        zone_detector.visualize_zone1_template_hits(
                            img, m.get("zone1_template_hits") or [], search_zone, search_x
                        ),
                    )
                    save_image(
                        debug_root / f"page_{p:02d}_zone1_fields_zone.jpg",
                        _draw_zone1_fields_band(img, m["zones"]["fields"]),
                    )

            # Dump full page metadata for debugging splits/logo confidence/template status.
            save_json(debug_root / "page_meta.json", page_meta)

        # STEP 3: Extraction (multi-document)
        docs = _group_pages_by_document(page_meta)
        doc_results_meta: List[Dict[str, Any]] = []
        field_regions: List[ExtractedRegion] = []
        table_regions: List[ExtractedRegion] = []

        with Timer("extract_all_documents") as t_extract:
            for did, doc_pages in sorted(docs.items(), key=lambda kv: int(kv[0])):
                first_pages = [m for m in doc_pages if bool(m.get("is_first_page", False))]
                if not first_pages:
                    # Should not happen (page 0 forced), but be safe.
                    first_pages = [doc_pages[0]]
                first_meta = first_pages[0]
                first_idx = int(first_meta.get("page_num", 0))

                log.info("Processing document %d: pages=%s first_page=%d", int(did), [int(m["page_num"]) + 1 for m in doc_pages], first_idx + 1)

                # Fields (only on first page)
                doc_field_regions: List[ExtractedRegion] = []
                if bool(first_meta.get("has_fields", False)):
                    field_zone = first_meta["zones"]["fields"]
                    prefix = f"doc{int(did)}_p{int(first_idx)+1}_"
                    fe = FieldExtractor(remove_dots=True)
                    doc_field_regions = fe.extract_fields(pages[first_idx], field_zone, debug=debug, region_id_prefix=prefix)
                    field_regions.extend(doc_field_regions)

                    if debug_root is not None:
                        dot_det = DotDetector()
                        dotted = dot_det.detect_dotted_lines(pages[first_idx], field_zone)
                        save_image(
                            debug_root / f"page_{int(first_idx)+1:02d}_dots_detected.jpg",
                            dot_det.visualize_detection(pages[first_idx], dotted, field_zone),
                        )
                        for r in doc_field_regions:
                            save_image(debug_root / f"{r.region_id}_raw.jpg", r.raw_image)
                            save_image(debug_root / f"{r.region_id}_preprocessed.jpg", r.image)

                # Tables (all pages in doc)
                td = TableDetector()
                te = TableExtractor(remove_grid=True)
                target_col: Optional[Tuple[int, int]] = None
                target_col_idx0: Optional[int] = None
                row_offset = 0
                doc_table_regions: List[ExtractedRegion] = []

                for m in doc_pages:
                    page_num = int(m.get("page_num", 0))
                    if not bool(m.get("has_table", False)):
                        continue
                    img = pages[page_num]
                    zone = m["zones"]["table"]
                    bbox = td.detect_table_boundary(img, zone)
                    struct = td.parse_table_structure(img, bbox)

                    if bool(m.get("is_first_page", False)):
                        target_col = struct.target_column
                        target_col_idx0 = int(struct.target_column_index)
                    elif target_col is not None and target_col_idx0 is not None:
                        struct = TableStructure(
                            bbox=struct.bbox,
                            grid_horizontal=struct.grid_horizontal,
                            grid_vertical=struct.grid_vertical,
                            target_column=target_col,
                            target_column_index=int(target_col_idx0),
                            rows=struct.rows,
                            cols=struct.cols,
                        )

                    page_cells = te.extract_target_column_cells(img, struct, page_num, row_offset=row_offset, skip_header_rows=1)
                    doc_table_regions.extend(page_cells)
                    row_offset += len(page_cells)

                    if debug_root is not None and page_cells:
                        overlay = img.copy()
                        x0, x1 = struct.target_column
                        cv2.rectangle(overlay, (int(x0), int(struct.bbox.y)), (int(x1), int(struct.bbox.y2)), (0, 255, 0), 6)
                        save_image(debug_root / f"page_{page_num+1:02d}_last_column_highlighted.jpg", overlay)
                        for r in page_cells[:12]:
                            save_image(debug_root / f"{r.region_id}_raw.jpg", r.raw_image)
                            save_image(debug_root / f"{r.region_id}_preprocessed.jpg", r.image)

                table_regions.extend(doc_table_regions)

                doc_results_meta.append(
                    {
                        "document_id": int(did),
                        "pages": [int(m["page_num"]) for m in doc_pages],
                        "first_page": int(first_idx),
                        "logo_detected": bool(first_meta.get("logo_detected", False)),
                        "logo_confidence": float(first_meta.get("logo_confidence", 0.0) or 0.0),
                        "extractions": {"fields": int(len(doc_field_regions)), "table_column_3": int(len(doc_table_regions))},
                    }
                )

        timings["step3_extract_all_documents_s"] = float(t_extract.dt or 0.0)

        # STEP 4: Batch OCR
        with Timer("ocr") as t:
            all_regions = field_regions + table_regions
            images = [r.image for r in all_regions]
            ids = [r.region_id for r in all_regions]
            ocr = OCRProcessor(provider=config.OCR_PROVIDER, languages=config.OCR_LANGUAGES)
            ocr_results = ocr.batch_ocr(images, ids)
        timings["step4_batch_ocr_s"] = float(t.dt or 0.0)

        # STEP 5: Structure + validate (per-document)
        with Timer("validate") as t:
            docs = _group_pages_by_document(page_meta)
            validator = Validator()
            documents_out: List[Dict[str, Any]] = []

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

                structured_doc = structure_results(f_regs, t_regs, ocr_results)
                validated_doc = validator.validate_results(structured_doc)
                documents_out.append(
                    {
                        "document_id": int(did),
                        "pages": [int(m["page_num"]) for m in doc_pages],
                        "first_page": int(first_idx),
                        **validated_doc,
                    }
                )

        timings["step5_validation_s"] = float(t.dt or 0.0)

        total_s = float(sum(timings.values()))
        result: Dict[str, Any] = {
            "metadata": {
                "pdf_path": str(pdf_path),
                "pages": int(len(pages)),
                "total_documents": int(len(doc_results_meta) or 1),
                "processing_time": round(total_s, 3),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "timings": {k: round(float(v), 3) for k, v in timings.items()},
                "step2_zone_detection_detail": step2_detail,
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
            r = process_form(str(p), str(out_dir), debug=bool(debug))
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


