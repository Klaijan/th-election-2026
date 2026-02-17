#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: PyMuPDF\n"
        "Install it into your environment, e.g.:\n"
        "  .venv/bin/python -m pip install PyMuPDF\n"
        "or:\n"
        "  python3 -m pip install PyMuPDF\n"
    ) from e


def _parse_rect(s: str) -> tuple[float, float, float, float]:
    """
    Parse "x0,y0,x1,y1" (commas) into 4 floats.
    """
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("rect must be 'x0,y0,x1,y1'")
    try:
        x0, y0, x1, y1 = (float(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError("rect values must be numbers") from e
    return x0, y0, x1, y1


def _rect_from_coords(
    *,
    rect: tuple[float, float, float, float],
    coord_space: str,
    page: fitz.Page,
    dpi: int,
) -> fitz.Rect:
    x0, y0, x1, y1 = rect

    if coord_space == "norm":
        w = float(page.rect.width)
        h = float(page.rect.height)
        return fitz.Rect(x0 * w, y0 * h, x1 * w, y1 * h)

    if coord_space == "pt":
        return fitz.Rect(x0, y0, x1, y1)

    if coord_space == "px":
        # Pixel coordinates assumed to be from a rendered image at the provided DPI.
        # Convert pixels -> PDF points.
        scale = dpi / 72.0
        return fitz.Rect(x0 / scale, y0 / scale, x1 / scale, y1 / scale)

    raise ValueError(f"Unknown coord_space: {coord_space}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Crop a rectangle from the first page of a PDF and save as an image (fast clip rendering)."
    )
    p.add_argument("--pdf", required=True, help="Input PDF file path.")
    p.add_argument(
        "--out",
        required=True,
        help=(
            "Output image path (e.g. crop.png) OR an output directory (e.g. crops/). "
            "If a directory is provided, a filename will be auto-generated."
        ),
    )
    p.add_argument(
        "--rect",
        type=_parse_rect,
        default=None,
        help=(
            "Crop rectangle as x0,y0,x1,y1. "
            "If omitted, defaults to full page width and 30%%..60%% of page height (normalized)."
        ),
    )
    p.add_argument(
        "--coord-space",
        choices=["norm", "pt", "px"],
        default="norm",
        help=(
            "Coordinate space for --rect: "
            "'norm' = normalized fractions of page width/height (0..1), "
            "'pt' = PDF points (72 points per inch), "
            "'px' = pixels at the specified --dpi."
        ),
    )
    p.add_argument("--page", type=int, default=0, help="0-based page index. Default: 0 (first page).")
    p.add_argument("--dpi", type=int, default=200, help="Render DPI. Default: 200.")
    p.add_argument(
        "--pdf-output",
        choices=["raster", "copy"],
        default="raster",
        help=(
            "When --out ends with .pdf: "
            "'raster' renders the crop and embeds it as an image (normalizes rotations; best for scans). "
            "'copy' copies original PDF content via show_pdf_page (keeps vectors but may preserve rotation quirks)."
        ),
    )
    args = p.parse_args(argv)

    pdf_path = Path(args.pdf)
    out_path = Path(args.out)
    # If user passed a directory (or a path without extension), auto-generate a filename.
    if str(args.out).endswith(("/", "\\")) or out_path.suffix == "":
        out_dir = out_path
        out_path = out_dir / f"{pdf_path.stem}_p{int(args.page)}.png"
    else:
        out_dir = out_path.parent

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, FileNotFoundError) as e:  # pragma: no cover
        raise SystemExit(
            f"Cannot create output directory: {out_dir}\n"
            f"Reason: {e}\n\n"
            "Tip: if you meant a path relative to the current folder, don't start it with '/'.\n"
            "Example:\n"
            "  --out data/sample/cropped/district.png\n"
        ) from e

    doc = fitz.open(str(pdf_path))
    try:
        if args.page < 0 or args.page >= doc.page_count:
            raise SystemExit(f"Page out of range: {args.page} (pages: {doc.page_count})")

        page = doc.load_page(args.page)
        if args.rect is None:
            # Default: full page width, 30%..60% of page height.
            rect = (0.0, 0.30, 1.0, 0.65)
            coord_space = "norm"
        else:
            rect = args.rect
            coord_space = args.coord_space

        clip = _rect_from_coords(rect=rect, coord_space=coord_space, page=page, dpi=int(args.dpi))

        # Ensure a valid rectangle (and intersect with page bounds).
        clip = clip.intersect(page.rect)
        if clip.is_empty or clip.width <= 0 or clip.height <= 0:
            raise SystemExit(f"Empty crop rect after intersecting with page bounds: {clip}")

        # Output selection:
        # - If --out ends with .pdf => write a new 1-page PDF containing only the clipped region.
        # - Otherwise => render the clipped region and save as an image (png/jpg/...).
        if out_path.suffix.lower() == ".pdf":
            out_pdf = fitz.open()
            try:
                new_page = out_pdf.new_page(width=clip.width, height=clip.height)
                dst = fitz.Rect(0, 0, clip.width, clip.height)

                if args.pdf_output == "copy":
                    new_page.show_pdf_page(dst, doc, args.page, clip=clip)
                else:
                    # Default: rasterize crop and embed image -> stable orientation across scanned PDFs.
                    scale = int(args.dpi) / 72.0
                    mat = fitz.Matrix(scale, scale)
                    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
                    new_page.insert_image(dst, pixmap=pix, keep_proportion=False)

                out_pdf.save(str(out_path))
            finally:
                out_pdf.close()
        else:
            scale = int(args.dpi) / 72.0
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
            # PyMuPDF chooses output format based on extension.
            pix.save(str(out_path))

        print(f"Cropped page {args.page} -> {out_path}")
        return 0
    finally:
        doc.close()


if __name__ == "__main__":
    raise SystemExit(main())


