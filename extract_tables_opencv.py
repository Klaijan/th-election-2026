#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional


def _require_deps():
    try:
        import fitz  # noqa: F401  # PyMuPDF
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: PyMuPDF\n"
            "Install deps:\n"
            "  python -m pip install -r requirements.txt\n"
        ) from e
    try:
        import cv2  # noqa: F401
        import numpy as np  # noqa: F401
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: opencv / numpy\n"
            "Install deps:\n"
            "  python -m pip install -r requirements.txt\n"
        ) from e


@dataclass(frozen=True)
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return max(0, int(self.x1) - int(self.x0))

    @property
    def h(self) -> int:
        return max(0, int(self.y1) - int(self.y0))

    @property
    def area(self) -> int:
        return int(self.w) * int(self.h)

    def clamp(self, *, width: int, height: int) -> "Box":
        x0 = max(0, min(int(self.x0), int(width)))
        x1 = max(0, min(int(self.x1), int(width)))
        y0 = max(0, min(int(self.y0), int(height)))
        y1 = max(0, min(int(self.y1), int(height)))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return Box(x0=x0, y0=y0, x1=x1, y1=y1)

    def pad(self, p: int) -> "Box":
        return Box(x0=self.x0 - p, y0=self.y0 - p, x1=self.x1 + p, y1=self.y1 + p)


@dataclass(frozen=True)
class TableExtraction:
    page_index: int
    table_index: int
    table_box_px: Box
    x_lines_px: list[int]
    y_lines_px: list[int]
    n_rows: int
    n_cols: int
    cells_px: list[list[Box]]  # row-major [r][c]


def iter_inputs(root: Path) -> list[Path]:
    exts = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
    if root.is_file():
        return [root] if root.suffix.lower() in exts else []
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in exts]


def _render_pdf_page_bgr(pdf: Path, *, page_index: int, dpi: int):
    import fitz
    import numpy as np
    import cv2

    doc = fitz.open(str(pdf))
    try:
        if page_index < 0 or page_index >= doc.page_count:
            raise ValueError(f"page out of range: {page_index} (pages: {doc.page_count})")
        pg = doc.load_page(int(page_index))
        mat = fitz.Matrix(float(dpi) / 72.0, float(dpi) / 72.0)
        pix = pg.get_pixmap(matrix=mat, alpha=False)
        # pix.samples is RGB bytes
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 3:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            # should not happen with alpha=False, but keep it robust
            bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return bgr
    finally:
        doc.close()


def _read_image_bgr(path: Path):
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"failed to read image: {path}")
    return img


def _binarize_for_lines(gray):
    """
    Returns a binary image where table lines are bright (255) on dark background (0).
    """
    import cv2

    # Invert so lines are emphasized (common for forms: dark lines on light background).
    inv = cv2.bitwise_not(gray)
    # Adaptive threshold tends to work better across scan lighting variations.
    bw = cv2.adaptiveThreshold(
        inv,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        35,
        -5,
    )
    return bw


def _detect_grid_masks(bw):
    """
    Given binarized image (lines are bright), return (vert_mask, horiz_mask, grid_mask).
    """
    import cv2
    import numpy as np

    h, w = bw.shape[:2]
    k = max(12, int(min(w, h) / 120))

    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))

    vert = cv2.erode(bw, vert_kernel, iterations=2)
    vert = cv2.dilate(vert, vert_kernel, iterations=2)

    horiz = cv2.erode(bw, horiz_kernel, iterations=2)
    horiz = cv2.dilate(horiz, horiz_kernel, iterations=2)

    grid = cv2.bitwise_or(vert, horiz)
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return vert, horiz, grid


def _find_table_boxes(grid_mask, *, min_area_frac: float, max_tables: int) -> list[Box]:
    import cv2

    h, w = grid_mask.shape[:2]
    min_area = float(min_area_frac) * float(w * h)
    contours, _hier = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[Box] = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        box = Box(x0=int(x), y0=int(y), x1=int(x + ww), y1=int(y + hh))
        if box.area < min_area:
            continue
        # Avoid super-thin artifacts.
        if box.w < 80 or box.h < 80:
            continue
        boxes.append(box)
    boxes.sort(key=lambda b: b.area, reverse=True)
    return boxes[: max(1, int(max_tables))]


def _line_positions_1d(proj, *, thr_frac: float) -> list[int]:
    """
    Convert a 1D projection into line positions (center of runs above threshold).
    """
    import numpy as np

    proj = np.asarray(proj, dtype=np.float32)
    if proj.size == 0:
        return []
    mx = float(proj.max()) if proj.size else 0.0
    if mx <= 0:
        return []
    thr = mx * float(thr_frac)
    on = proj >= thr

    idxs: list[int] = []
    i = 0
    n = int(on.size)
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i + 1
        while j < n and on[j]:
            j += 1
        center = int(round((i + (j - 1)) / 2.0))
        idxs.append(center)
        i = j
    # De-dup very close positions.
    idxs.sort()
    merged: list[int] = []
    for x in idxs:
        if not merged or abs(x - merged[-1]) > 3:
            merged.append(x)
    return merged


def _extract_cells_from_lines(x_lines: list[int], y_lines: list[int], *, min_cell_w: int, min_cell_h: int) -> list[list[Box]]:
    x_lines = sorted(set(int(x) for x in x_lines))
    y_lines = sorted(set(int(y) for y in y_lines))
    cells: list[list[Box]] = []
    if len(x_lines) < 2 or len(y_lines) < 2:
        return cells
    for r in range(len(y_lines) - 1):
        row: list[Box] = []
        for c in range(len(x_lines) - 1):
            x0, x1 = x_lines[c], x_lines[c + 1]
            y0, y1 = y_lines[r], y_lines[r + 1]
            box = Box(x0=x0, y0=y0, x1=x1, y1=y1)
            if box.w < min_cell_w or box.h < min_cell_h:
                # Keep structure but mark as tiny; downstream can ignore if needed.
                row.append(box)
            else:
                row.append(box)
        cells.append(row)
    return cells


def extract_tables_from_bgr(
    bgr,
    *,
    max_tables: int,
    min_table_area_frac: float,
    pad_px: int,
    line_thr_frac: float,
):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bw = _binarize_for_lines(gray)
    vert, horiz, grid = _detect_grid_masks(bw)

    table_boxes = _find_table_boxes(grid, min_area_frac=min_table_area_frac, max_tables=max_tables)
    h, w = gray.shape[:2]

    extractions: list[tuple[Box, dict]] = []
    for tbox in table_boxes:
        box = tbox.pad(int(pad_px)).clamp(width=w, height=h)

        # Restrict to ROI for grid line estimation.
        v_roi = vert[box.y0 : box.y1, box.x0 : box.x1]
        h_roi = horiz[box.y0 : box.y1, box.x0 : box.x1]

        # Projections: how much "line ink" per column/row
        vx = np.sum(v_roi > 0, axis=0)
        vy = np.sum(h_roi > 0, axis=1)

        x_lines_local = _line_positions_1d(vx, thr_frac=float(line_thr_frac))
        y_lines_local = _line_positions_1d(vy, thr_frac=float(line_thr_frac))

        # Convert to absolute page coords.
        x_lines = [int(box.x0 + x) for x in x_lines_local]
        y_lines = [int(box.y0 + y) for y in y_lines_local]

        cells = _extract_cells_from_lines(x_lines, y_lines, min_cell_w=12, min_cell_h=12)
        n_rows = max(0, len(y_lines) - 1)
        n_cols = max(0, len(x_lines) - 1)

        extractions.append(
            (
                box,
                {
                    "x_lines": x_lines,
                    "y_lines": y_lines,
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "cells": cells,
                },
            )
        )

    debug = {
        "gray": gray,
        "bw": bw,
        "vert": vert,
        "horiz": horiz,
        "grid": grid,
    }
    return extractions, debug


def _write_debug_images(debug_dir: Path, *, bgr, debug: dict, tables: list[Box]) -> None:
    import cv2
    import numpy as np

    debug_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(debug_dir / "page.png"), bgr)
    cv2.imwrite(str(debug_dir / "gray.png"), debug["gray"])
    cv2.imwrite(str(debug_dir / "bw.png"), debug["bw"])
    cv2.imwrite(str(debug_dir / "vert.png"), debug["vert"])
    cv2.imwrite(str(debug_dir / "horiz.png"), debug["horiz"])
    cv2.imwrite(str(debug_dir / "grid.png"), debug["grid"])

    overlay = bgr.copy()
    for i, box in enumerate(tables):
        cv2.rectangle(overlay, (box.x0, box.y0), (box.x1, box.y1), (0, 0, 255), 6)
        cv2.putText(
            overlay,
            f"table_{i}",
            (box.x0 + 10, max(0, box.y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 0, 255),
            4,
            lineType=cv2.LINE_AA,
        )
    cv2.imwrite(str(debug_dir / "tables_overlay.png"), overlay)

    # Tiny composite for quick inspection
    def _norm(x):
        if x.ndim == 2:
            return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
        return x

    grid_rgb = _norm(debug["grid"])
    bw_rgb = _norm(debug["bw"])
    comp = np.concatenate([bw_rgb, grid_rgb], axis=1)
    cv2.imwrite(str(debug_dir / "bw__grid.png"), comp)


def _save_cells(out_dir: Path, *, bgr, cells: list[list[Box]]) -> None:
    import cv2

    cells_dir = out_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    for r, row in enumerate(cells):
        for c, box in enumerate(row):
            crop = bgr[box.y0 : box.y1, box.x0 : box.x1]
            if crop.size == 0:
                continue
            cv2.imwrite(str(cells_dir / f"cell_r{r:03d}_c{c:03d}.png"), crop)


def _process_one(
    *,
    in_path: Path,
    out_root: Path,
    dpi: int,
    page_index: Optional[int],
    max_tables: int,
    min_table_area_frac: float,
    pad_px: int,
    line_thr_frac: float,
    save_cells: bool,
    debug: bool,
) -> dict:
    import cv2

    rel = in_path.name
    out_dir = out_root / in_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[TableExtraction] = []

    if in_path.suffix.lower() == ".pdf":
        # Only process a single page by default (these sample forms are 1-page PDFs).
        pages: Iterable[int] = [int(page_index or 0)]
        for pidx in pages:
            bgr = _render_pdf_page_bgr(in_path, page_index=pidx, dpi=int(dpi))
            extractions, dbg = extract_tables_from_bgr(
                bgr,
                max_tables=int(max_tables),
                min_table_area_frac=float(min_table_area_frac),
                pad_px=int(pad_px),
                line_thr_frac=float(line_thr_frac),
            )

            tables = [box for (box, _meta) in extractions]
            page_out = out_dir / f"page_{pidx:03d}"
            page_out.mkdir(parents=True, exist_ok=True)

            if debug:
                _write_debug_images(page_out / "_debug", bgr=bgr, debug=dbg, tables=tables)

            for tidx, (tbox, meta) in enumerate(extractions):
                table_out = page_out / f"table_{tidx:02d}"
                table_out.mkdir(parents=True, exist_ok=True)
                table_bgr = bgr[tbox.y0 : tbox.y1, tbox.x0 : tbox.x1]
                cv2.imwrite(str(table_out / "table.png"), table_bgr)

                x_lines = meta["x_lines"]
                y_lines = meta["y_lines"]
                cells: list[list[Box]] = meta["cells"]
                if save_cells and cells:
                    _save_cells(table_out, bgr=bgr, cells=cells)

                results.append(
                    TableExtraction(
                        page_index=int(pidx),
                        table_index=int(tidx),
                        table_box_px=tbox,
                        x_lines_px=list(map(int, x_lines)),
                        y_lines_px=list(map(int, y_lines)),
                        n_rows=int(meta["n_rows"]),
                        n_cols=int(meta["n_cols"]),
                        cells_px=cells,
                    )
                )
    else:
        bgr = _read_image_bgr(in_path)
        extractions, dbg = extract_tables_from_bgr(
            bgr,
            max_tables=int(max_tables),
            min_table_area_frac=float(min_table_area_frac),
            pad_px=int(pad_px),
            line_thr_frac=float(line_thr_frac),
        )
        tables = [box for (box, _meta) in extractions]
        if debug:
            _write_debug_images(out_dir / "_debug", bgr=bgr, debug=dbg, tables=tables)
        for tidx, (tbox, meta) in enumerate(extractions):
            table_out = out_dir / f"table_{tidx:02d}"
            table_out.mkdir(parents=True, exist_ok=True)
            table_bgr = bgr[tbox.y0 : tbox.y1, tbox.x0 : tbox.x1]
            cv2.imwrite(str(table_out / "table.png"), table_bgr)

            x_lines = meta["x_lines"]
            y_lines = meta["y_lines"]
            cells: list[list[Box]] = meta["cells"]
            if save_cells and cells:
                _save_cells(table_out, bgr=bgr, cells=cells)

            results.append(
                TableExtraction(
                    page_index=0,
                    table_index=int(tidx),
                    table_box_px=tbox,
                    x_lines_px=list(map(int, x_lines)),
                    y_lines_px=list(map(int, y_lines)),
                    n_rows=int(meta["n_rows"]),
                    n_cols=int(meta["n_cols"]),
                    cells_px=cells,
                )
            )

    out_json = out_dir / "tables.json"
    payload = {
        "input": str(in_path),
        "dpi": int(dpi),
        "tables": [
            {
                **asdict(t),
                "table_box_px": asdict(t.table_box_px),
                "cells_px": [[asdict(b) for b in row] for row in t.cells_px],
            }
            for t in results
        ],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"input": str(in_path), "out_dir": str(out_dir), "n_tables": len(results)}


def main(argv: list[str] | None = None) -> int:
    _require_deps()

    p = argparse.ArgumentParser(
        description=(
            "Extract table regions (and optionally grid cells) from PDFs/images using OpenCV.\n"
            "Writes crops + a tables.json with bounding boxes.\n\n"
            "Example:\n"
            "  python extract_tables_opencv.py --input data/sample/cropped --out-root data/sample/output/opencv_tables --debug\n"
        )
    )
    p.add_argument("--input", required=True, help="Input PDF/image file OR a directory (recursive).")
    p.add_argument("--out-root", required=True, help="Output directory root.")
    p.add_argument("--dpi", type=int, default=220, help="PDF render DPI. Default: 220.")
    p.add_argument("--page", type=int, default=0, help="0-based page index for PDFs. Default: 0.")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2), help="Concurrency. Default: ~half CPUs.")
    p.add_argument("--max-tables", type=int, default=2, help="Max tables to keep per page (largest first). Default: 2.")
    p.add_argument(
        "--min-table-area-frac",
        type=float,
        default=0.04,
        help="Min table area as fraction of page area. Default: 0.04.",
    )
    p.add_argument("--pad-px", type=int, default=8, help="Padding around detected table bounding box. Default: 8.")
    p.add_argument(
        "--line-thr-frac",
        type=float,
        default=0.55,
        help="Projection threshold as fraction of max to detect grid lines. Default: 0.55.",
    )
    p.add_argument("--save-cells", action="store_true", help="Also write cell crops (can be many files).")
    p.add_argument("--debug", action="store_true", help="Write intermediate images for debugging.")
    args = p.parse_args(argv)

    in_root = Path(args.input)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    inputs = iter_inputs(in_root)
    if not inputs:
        raise SystemExit(f"No inputs found under: {in_root}")

    failures: list[tuple[Path, str]] = []
    done = 0

    with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = [
            ex.submit(
                _process_one,
                in_path=path,
                out_root=out_root,
                dpi=int(args.dpi),
                page_index=int(args.page),
                max_tables=int(args.max_tables),
                min_table_area_frac=float(args.min_table_area_frac),
                pad_px=int(args.pad_px),
                line_thr_frac=float(args.line_thr_frac),
                save_cells=bool(args.save_cells),
                debug=bool(args.debug),
            )
            for path in inputs
        ]

        for fut in as_completed(futs):
            try:
                r = fut.result()
                done += 1
                print(f"[{done}/{len(inputs)}] {r['input']} -> {r['n_tables']} table(s)")
            except Exception as e:
                # Best effort error reporting.
                failures.append((Path("?"), str(e)))

    if failures:
        print(f"Completed with {len(failures)} failure(s):")
        for _p, msg in failures[:50]:
            print(f"- {msg}")
        return 1

    print(f"Done. Wrote outputs under: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


