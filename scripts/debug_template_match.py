#!/usr/bin/env python3
"""
Debug template detection on a PDF page by running OpenCV template matching and
writing out visualizations (ROI, heatmap, best-hit overlay).
Designed to answer: "Why didn't template_4 detect even though it was cropped from this PDF?"
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

try:
    import cv2  # type: ignore
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: OpenCV (cv2)\n\n"
        "Install project dependencies, e.g.:\n"
        "  python3 -m pip install -r requirements.txt\n\n"
        "Or just OpenCV:\n"
        "  python3 -m pip install opencv-python-headless\n"
    ) from e

try:
    import numpy as np  # type: ignore
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: numpy\n\n"
        "Install project dependencies, e.g.:\n"
        "  python3 -m pip install -r requirements.txt\n"
    ) from e


Backend = Literal["pdf2image", "pymupdf", "auto", "both"]


@dataclass(frozen=True)
class Match:
    score: float
    x: int
    y: int
    w: int
    h: int
    scale: float
    method: str
    mode: str


def _parse_csv(s: str) -> list[str]:
    return [p.strip() for p in (s or "").split(",") if p.strip()]


def _method_from_name(name: str) -> tuple[int, str]:
    n = str(name or "").strip()
    if n == "TM_CCOEFF_NORMED":
        return (cv2.TM_CCOEFF_NORMED, "TM_CCOEFF_NORMED")
    if n == "TM_SQDIFF_NORMED":
        return (cv2.TM_SQDIFF_NORMED, "TM_SQDIFF_NORMED")
    raise ValueError(f"Unknown method: {name}")


def _parse_scales(s: str) -> list[float]:
    """
    Accept either:
      - "0.9,1.0,1.1"
      - "start:stop:step" (inclusive-ish; stop is included if we land on it)
    """
    s = s.strip()
    if ":" in s:
        parts = [p.strip() for p in s.split(":")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("scales must be 'a,b,c' or 'start:stop:step'")
        a, b, step = (float(p) for p in parts)
        if step <= 0:
            raise argparse.ArgumentTypeError("scale step must be > 0")
        out: list[float] = []
        x = a
        # avoid float infinite loops
        for _ in range(2000):
            if x > b + 1e-9:
                break
            out.append(float(x))
            x = x + step
        if not out:
            raise argparse.ArgumentTypeError("scale range produced no values")
        return out
    out = [float(p.strip()) for p in s.split(",") if p.strip()]
    if not out:
        raise argparse.ArgumentTypeError("no scales provided")
    return out


def _parse_range_frac(s: str) -> tuple[float, float]:
    """
    Parse "a:b" into (a,b), clamped to [0,1].
    """
    parts = [p.strip() for p in (s or "").split(":")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must be 'a:b' where a,b are fractions 0..1")
    try:
        a, b = float(parts[0]), float(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError("range values must be numbers") from e
    a = max(0.0, min(a, 1.0))
    b = max(0.0, min(b, 1.0))
    if b < a:
        a, b = b, a
    return (float(a), float(b))


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _imwrite(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def _render_page_pdf2image(pdf_path: Path, *, page_index: int, dpi: int) -> np.ndarray:
    try:
        from pdf2image import convert_from_path  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pdf2image is not available (pip install pdf2image + install poppler)") from e

    pil_pages = convert_from_path(str(pdf_path), dpi=int(dpi), first_page=page_index + 1, last_page=page_index + 1)
    if not pil_pages:
        raise RuntimeError("pdf2image returned no pages")
    rgb = np.array(pil_pages[0].convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def _render_page_pymupdf(pdf_path: Path, *, page_index: int, dpi: int) -> np.ndarray:
    try:
        import fitz  # PyMuPDF
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyMuPDF is not available (pip install PyMuPDF)") from e

    doc = fitz.open(str(pdf_path))
    try:
        if page_index < 0 or page_index >= int(doc.page_count):
            raise RuntimeError(f"page_index out of range: {page_index} (pages={doc.page_count})")
        pg = doc.load_page(int(page_index))
        mat = fitz.Matrix(float(dpi) / 72.0, float(dpi) / 72.0)
        pix = pg.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 3:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return bgr
    finally:
        doc.close()


def _pdf_page_count(pdf_path: Path) -> int:
    """
    Best-effort page count, used when --page is omitted (process all pages).
    """
    try:
        import fitz  # PyMuPDF
    except Exception:
        fitz = None  # type: ignore[assignment]

    if fitz is not None:
        doc = fitz.open(str(pdf_path))
        try:
            return int(doc.page_count)
        finally:
            doc.close()

    # Fallback: pypdf (often present via other tooling)
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Cannot determine PDF page count (need PyMuPDF or pypdf installed). "
            "Install project deps: python3 -m pip install -r requirements.txt"
        ) from e

    r = PdfReader(str(pdf_path))
    return int(len(getattr(r, "pages", []) or []))


def _roi_from_fracs(img: np.ndarray, *, x_frac: float, y_frac: float) -> tuple[np.ndarray, dict]:
    # Backwards-compat: interpret x_frac/y_frac as (x1_frac,y1_frac) with (x0_frac,y0_frac)=(0,0)
    return _roi_from_box_fracs(img, x0_frac=0.0, x1_frac=float(x_frac), y0_frac=0.0, y1_frac=float(y_frac))


def _roi_from_box_fracs(
    img: np.ndarray, *, x0_frac: float, x1_frac: float, y0_frac: float, y1_frac: float
) -> tuple[np.ndarray, dict]:
    h, w = img.shape[:2]
    x0f = max(0.0, min(float(x0_frac), 1.0))
    x1f = max(0.0, min(float(x1_frac), 1.0))
    y0f = max(0.0, min(float(y0_frac), 1.0))
    y1f = max(0.0, min(float(y1_frac), 1.0))
    if x1f < x0f:
        x0f, x1f = x1f, x0f
    if y1f < y0f:
        y0f, y1f = y1f, y0f
    x0 = int(round(float(w) * x0f))
    x1 = int(round(float(w) * x1f))
    y0 = int(round(float(h) * y0f))
    y1 = int(round(float(h) * y1f))
    x0 = max(0, min(x0, w - 1))
    x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))
    roi = img[y0:y1, x0:x1]
    return roi, {"x0": int(x0), "x1": int(x1), "y0": int(y0), "y1": int(y1), "w": int(w), "h": int(h)}


def _prep(gray: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return gray
    if mode == "eq":
        return cv2.equalizeHist(gray)
    if mode == "blur":
        return cv2.GaussianBlur(gray, (3, 3), 0)
    if mode == "otsu":
        _t, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bw
    if mode == "canny":
        # fixed thresholds; good enough for debugging
        return cv2.Canny(gray, 60, 180)
    raise ValueError(f"Unknown mode: {mode}")


def _match_best(
    roi_gray: np.ndarray,
    tmpl_gray: np.ndarray,
    *,
    scales: Iterable[float],
    method: int,
    method_name: str,
    mode: str,
) -> Match | None:
    best: Match | None = None
    for s in scales:
        if not (0.05 <= float(s) <= 10.0) or math.isnan(float(s)) or math.isinf(float(s)):
            continue
        tw = max(4, int(round(tmpl_gray.shape[1] * float(s))))
        th = max(4, int(round(tmpl_gray.shape[0] * float(s))))
        if tw >= roi_gray.shape[1] or th >= roi_gray.shape[0]:
            continue
        t = cv2.resize(
            tmpl_gray,
            (int(tw), int(th)),
            interpolation=cv2.INTER_AREA if float(s) < 1 else cv2.INTER_CUBIC,
        )
        res = cv2.matchTemplate(roi_gray, t, method)
        minv, maxv, minl, maxl = cv2.minMaxLoc(res)

        if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            score = float(1.0 - float(minv))  # flip so higher is better
            x, y = int(minl[0]), int(minl[1])
        else:
            score = float(maxv)
            x, y = int(maxl[0]), int(maxl[1])

        if best is None or float(score) > float(best.score):
            best = Match(
                score=float(score),
                x=int(x),
                y=int(y),
                w=int(tw),
                h=int(th),
                scale=float(s),
                method=str(method_name),
                mode=str(mode),
            )
    return best


def _heatmap_uint8(res: np.ndarray, *, method: int) -> np.ndarray:
    r = res.copy()
    if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
        # smaller is better -> invert
        r = 1.0 - r
    r = np.clip(r, 0.0, 1.0)
    r = (r * 255.0).astype(np.uint8)
    return r


def _run_one_backend(
    *,
    backend: Literal["pdf2image", "pymupdf"],
    pdf_path: Path,
    template_path: Path,
    out_dir: Path,
    page_index: int,
    dpi: int,
    x0_frac: float,
    x1_frac: float,
    y0_frac: float,
    y1_frac: float,
    scales: list[float],
    thr: float,
    modes: list[str],
    methods: list[str],
    save_best: bool,
    save_rejected_best: bool,
    downscale: float,
) -> int:
    if backend == "pdf2image":
        page_bgr = _render_page_pdf2image(pdf_path, page_index=page_index, dpi=dpi)
    else:
        page_bgr = _render_page_pymupdf(pdf_path, page_index=page_index, dpi=dpi)

    roi_bgr, roi_meta = _roi_from_box_fracs(
        page_bgr,
        x0_frac=float(x0_frac),
        x1_frac=float(x1_frac),
        y0_frac=float(y0_frac),
        y1_frac=float(y1_frac),
    )
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    tmpl_gray = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if tmpl_gray is None or tmpl_gray.size == 0:
        raise SystemExit(f"Failed to load template: {template_path}")

    # Save inputs for inspection
    _imwrite(out_dir / f"{backend}_p{page_index+1:02d}_page.jpg", page_bgr)
    _imwrite(out_dir / f"{backend}_p{page_index+1:02d}_roi.jpg", roi_bgr)
    _imwrite(out_dir / f"{backend}_template.jpg", tmpl_gray)

    ds = max(0.05, min(float(downscale), 1.0))
    if ds != 1.0:
        roi_gray = cv2.resize(
            roi_gray,
            (max(32, int(round(roi_gray.shape[1] * ds))), max(32, int(round(roi_gray.shape[0] * ds)))),
            interpolation=cv2.INTER_AREA,
        )
        tmpl_gray = cv2.resize(
            tmpl_gray,
            (max(8, int(round(tmpl_gray.shape[1] * ds))), max(8, int(round(tmpl_gray.shape[0] * ds)))),
            interpolation=cv2.INTER_AREA,
        )

    # Modes/methods help diagnose contrast / anti-alias issues.
    # IMPORTANT:
    # - The production pipeline uses TM_CCOEFF_NORMED on raw grayscale.
    # - TM_SQDIFF_NORMED can look "very confident" on non-matching, low-texture areas,
    #   especially after blur. Keep it opt-in.
    methods_parsed: list[tuple[int, str]] = [_method_from_name(m) for m in methods]

    rows: list[Match] = []
    for mode in modes:
        r = _prep(roi_gray, mode)
        t = _prep(tmpl_gray, mode)
        for m, name in methods_parsed:
            hit = _match_best(r, t, scales=scales, method=m, method_name=name, mode=mode)
            if hit is not None:
                rows.append(hit)

    rows.sort(key=lambda x: float(x.score), reverse=True)
    if not rows:
        print(f"[{backend}] No valid matches (template bigger than ROI at all scales?)")
        return 2

    # Print top hits
    print(
        f"\n[{backend}] pdf={pdf_path.name} page={page_index+1} dpi={dpi} "
        f"roi=({roi_meta['x0']},{roi_meta['y0']})..({roi_meta['x1']},{roi_meta['y1']}) downscale={ds}"
    )
    print(f"[{backend}] template={template_path.name} size={tmpl_gray.shape[1]}x{tmpl_gray.shape[0]} px")
    print(f"[{backend}] scales: {scales[0]}..{scales[-1]} (n={len(scales)})  thr(ref)={thr}")
    for i, h in enumerate(rows[:10]):
        ok = "OK" if float(h.score) >= float(thr) else "no"
        px = int(round(float(h.x) / ds)) if ds != 1.0 else int(h.x)
        py = int(round(float(h.y) / ds)) if ds != 1.0 else int(h.y)
        pw = int(round(float(h.w) / ds)) if ds != 1.0 else int(h.w)
        ph = int(round(float(h.h) / ds)) if ds != 1.0 else int(h.h)
        print(
            f"[{backend}] #{i+1:02d} score={h.score:.4f} ({ok})  mode={h.mode:5s}  method={h.method:16s}  "
            f"scale={h.scale:.3f}  xy=({px},{py})  wh=({pw},{ph})"
        )

    # Visualize only the best hit overall (across modes/methods)
    best = rows[0]

    if not bool(save_best):
        print(f"[{backend}] --no-save-best set; skipping overlay/heatmap saves")
        return 0

    is_ok = float(best.score) >= float(thr)
    if (not is_ok) and (not bool(save_rejected_best)):
        print(
            f"[{backend}] best score {best.score:.4f} < thr {thr:.4f} -> rejected; --no-save-rejected-best set; skipping overlay/heatmap saves"
        )
        return 0

    # Recompute res for heatmap (only for raw+best.method to keep it interpretable)
    best_method = cv2.TM_CCOEFF_NORMED if best.method == "TM_CCOEFF_NORMED" else cv2.TM_SQDIFF_NORMED
    roi_for_map = _prep(roi_gray, "raw")
    tmpl_for_map = _prep(tmpl_gray, "raw")
    tw = max(4, int(round(tmpl_for_map.shape[1] * float(best.scale))))
    th = max(4, int(round(tmpl_for_map.shape[0] * float(best.scale))))
    tmpl_scaled = cv2.resize(
        tmpl_for_map,
        (int(tw), int(th)),
        interpolation=cv2.INTER_AREA if float(best.scale) < 1 else cv2.INTER_CUBIC,
    )
    res = cv2.matchTemplate(roi_for_map, tmpl_scaled, best_method)
    hm = _heatmap_uint8(res, method=best_method)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm_color = cv2.resize(hm_color, (roi_bgr.shape[1], roi_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay = roi_bgr.copy()
    bx = int(round(float(best.x) / ds)) if ds != 1.0 else int(best.x)
    by = int(round(float(best.y) / ds)) if ds != 1.0 else int(best.y)
    bw = int(round(float(best.w) / ds)) if ds != 1.0 else int(best.w)
    bh = int(round(float(best.h) / ds)) if ds != 1.0 else int(best.h)
    cv2.rectangle(
        overlay,
        (int(bx), int(by)),
        (int(bx + bw), int(by + bh)),
        (255, 0, 255),
        3,
    )
    cv2.putText(
        overlay,
        f"best: score={best.score:.3f} ({'OK' if is_ok else 'REJECTED'})  thr={thr:.3f}  scale={best.scale:.3f}  mode={best.mode}  method={best.method}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 255),
        2,
        lineType=cv2.LINE_AA,
    )

    blend = cv2.addWeighted(roi_bgr, 0.55, hm_color, 0.45, 0.0)

    _imwrite(out_dir / f"{backend}_p{page_index+1:02d}_best_overlay.jpg", overlay)
    _imwrite(out_dir / f"{backend}_p{page_index+1:02d}_heatmap.jpg", hm_color)
    _imwrite(out_dir / f"{backend}_p{page_index+1:02d}_blend.jpg", blend)

    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Debug template matching on a PDF using template_4 (or any template).")
    ap.add_argument(
        "--pdf",
        default=str(Path("data/sample/district/5.pdf")),
        help="Input PDF path. Default: data/sample/district/5.pdf",
    )
    ap.add_argument(
        "--template",
        default=str(Path("extract_handwritten_numbers/templates/template_4.png")),
        help="Template image (grayscale PNG). Default: extract_handwritten_numbers/templates/template_4.png",
    )
    ap.add_argument(
        "--page",
        type=int,
        default=None,
        help="0-based page index. If omitted, process ALL pages.",
    )
    ap.add_argument("--dpi", type=int, default=400, help="Render DPI. Default: 400 (matches pipeline default)")
    ap.add_argument(
        "--backend",
        choices=["pdf2image", "pymupdf", "auto", "both"],
        default="auto",
        help="PDF render backend. 'auto' behaves like the pipeline (pdf2image then fallback). 'both' compares outputs.",
    )
    ap.add_argument(
        "--x-frac",
        type=float,
        default=0.5,
        help="Back-compat: ROI x1 fraction with x0=0. Prefer --x0-frac/--x1-frac. Default: 0.5",
    )
    ap.add_argument(
        "--y-frac",
        type=float,
        default=0.65,
        help="Back-compat: ROI y1 fraction with y0=0. Prefer --y0-frac/--y1-frac. Default: 0.65",
    )
    ap.add_argument("--x0-frac", type=float, default=None, help="ROI x0 fraction. Default: 0.0")
    ap.add_argument("--x1-frac", type=float, default=None, help="ROI x1 fraction. Default: --x-frac")
    ap.add_argument("--y0-frac", type=float, default=None, help="ROI y0 fraction. Default: 0.0")
    ap.add_argument("--y1-frac", type=float, default=None, help="ROI y1 fraction. Default: --y-frac")
    ap.add_argument(
        "--scales",
        type=_parse_scales,
        default=_parse_scales("0.50:2.50:0.05"),
        help="Scales to test, e.g. '0.9,1.0,1.1' or '0.5:2.5:0.05'. Default: 0.50:2.50:0.05",
    )
    ap.add_argument(
        "--modes",
        default="raw",
        help="Comma-separated preprocessing modes to try: raw,eq,blur,otsu,canny. Default: raw (matches pipeline).",
    )
    ap.add_argument(
        "--methods",
        default="TM_CCOEFF_NORMED",
        help="Comma-separated matchTemplate methods to try: TM_CCOEFF_NORMED,TM_SQDIFF_NORMED. Default: TM_CCOEFF_NORMED (matches pipeline).",
    )
    ap.add_argument(
        "--thr",
        type=float,
        default=0.60,
        help="Reference threshold to label hits as OK/no (pipeline uses 0.75).",
    )
    ap.add_argument(
        "--downscale",
        type=float,
        default=1.0,
        help="Downscale ROI+template before matching (speed). 0.25 matches LogoDetector. Default: 1.0",
    )
    ap.add_argument(
        "--preset",
        choices=["none", "logo", "region"],
        default="none",
        help="Preset to match pipeline defaults. 'logo' matches LogoDetector ROI+downscale. 'region' matches region_begin/end matcher ROI.",
    )
    ap.add_argument(
        "--save-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to save best-match visualizations (best_overlay/heatmap/blend). "
            "Defaults to true. Use --no-save-best to disable."
        ),
    )
    ap.add_argument(
        "--save-rejected-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When best score is below --thr, still save best-match visualizations. "
            "Defaults to true. Use --no-save-rejected-best to save overlays only when best score passes --thr."
        ),
    )
    ap.add_argument(
        "--out",
        default=str(Path("data/output/template_match_debug")),
        help="Output directory for debug images. Default: data/output/template_match_debug/",
    )

    args = ap.parse_args(argv)

    # Apply presets only when user didn't explicitly provide corresponding flags.
    argv_tokens = set(sys.argv[1:] if argv is None else (argv or []))

    def _flag_present(name: str) -> bool:
        return any(t == name or str(t).startswith(name + "=") for t in argv_tokens)

    if str(args.preset) == "logo":
        if not (_flag_present("--y-frac") or _flag_present("--y1-frac")):
            args.y_frac = 0.20
        if not _flag_present("--x0-frac"):
            args.x0_frac = 0.20
        if not (_flag_present("--x-frac") or _flag_present("--x1-frac")):
            args.x_frac = 0.80
        if not _flag_present("--downscale"):
            args.downscale = 0.25
        if not _flag_present("--methods"):
            args.methods = "TM_CCOEFF_NORMED"
        if not _flag_present("--modes"):
            args.modes = "raw"
        if not _flag_present("--scales"):
            args.scales = _parse_scales("0.50:2.50:0.05")
        if not _flag_present("--thr"):
            args.thr = 0.70

    if str(args.preset) == "region":
        if not (_flag_present("--y-frac") or _flag_present("--y1-frac")):
            args.y_frac = 0.65
        if not _flag_present("--x0-frac"):
            args.x0_frac = 0.00
        if not (_flag_present("--x-frac") or _flag_present("--x1-frac")):
            args.x_frac = 1.00
        if not _flag_present("--downscale"):
            args.downscale = 1.00
        if not _flag_present("--methods"):
            args.methods = "TM_CCOEFF_NORMED"
        if not _flag_present("--modes"):
            args.modes = "raw"
        if not _flag_present("--scales"):
            args.scales = _parse_scales("0.50:2.50:0.05")
    pdf_path = Path(args.pdf)
    template_path = Path(args.template)
    out_dir = _ensure_dir(Path(args.out))

    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")
    if not template_path.exists():
        raise SystemExit(f"Template not found: {template_path}")

    # Validate methods/modes early (kept minimal; this is a debug tool).
    wanted_modes = _parse_csv(str(args.modes))
    wanted_methods = _parse_csv(str(args.methods))
    valid_modes = {"raw", "eq", "blur", "otsu", "canny"}
    valid_methods = {"TM_CCOEFF_NORMED", "TM_SQDIFF_NORMED"}
    bad_modes = [m for m in wanted_modes if m not in valid_modes]
    bad_methods = [m for m in wanted_methods if m not in valid_methods]
    if bad_modes:
        raise SystemExit(f"Unknown --modes entries: {bad_modes}. Valid: {sorted(valid_modes)}")
    if bad_methods:
        raise SystemExit(f"Unknown --methods entries: {bad_methods}. Valid: {sorted(valid_methods)}")

    back: Backend = args.backend

    x0_frac = 0.0 if args.x0_frac is None else float(args.x0_frac)
    y0_frac = 0.0 if args.y0_frac is None else float(args.y0_frac)
    x1_frac = float(args.x_frac) if args.x1_frac is None else float(args.x1_frac)
    y1_frac = float(args.y_frac) if args.y1_frac is None else float(args.y1_frac)

    if args.page is None:
        page_count = _pdf_page_count(pdf_path)
        pages_to_run = list(range(int(page_count)))
    else:
        pages_to_run = [int(args.page)]

    # Match pipeline behavior: try pdf2image, fallback to pymupdf.
    if back == "auto":
        rc = 0
        for page_index in pages_to_run:
            try:
                rc = rc or int(
                    _run_one_backend(
                        backend="pdf2image",
                        pdf_path=pdf_path,
                        template_path=template_path,
                        out_dir=out_dir,
                        page_index=int(page_index),
                        dpi=int(args.dpi),
                        x0_frac=float(x0_frac),
                        x1_frac=float(x1_frac),
                        y0_frac=float(y0_frac),
                        y1_frac=float(y1_frac),
                        scales=list(args.scales),
                        thr=float(args.thr),
                        modes=wanted_modes,
                        methods=wanted_methods,
                        save_best=bool(args.save_best),
                        save_rejected_best=bool(args.save_rejected_best),
                        downscale=float(args.downscale),
                    )
                )
            except Exception as e:
                print(f"[auto] pdf2image failed on page {int(page_index)+1}: {e}\n[auto] falling back to PyMuPDF...")
                rc = rc or int(
                    _run_one_backend(
                        backend="pymupdf",
                        pdf_path=pdf_path,
                        template_path=template_path,
                        out_dir=out_dir,
                        page_index=int(page_index),
                        dpi=int(args.dpi),
                        x0_frac=float(x0_frac),
                        x1_frac=float(x1_frac),
                        y0_frac=float(y0_frac),
                        y1_frac=float(y1_frac),
                        scales=list(args.scales),
                        thr=float(args.thr),
                        modes=wanted_modes,
                        methods=wanted_methods,
                        save_best=bool(args.save_best),
                        save_rejected_best=bool(args.save_rejected_best),
                        downscale=float(args.downscale),
                    )
                )
        return int(rc)

    if back == "both":
        rc = 0
        for page_index in pages_to_run:
            rc1 = _run_one_backend(
                backend="pdf2image",
                pdf_path=pdf_path,
                template_path=template_path,
                out_dir=out_dir,
                page_index=int(page_index),
                dpi=int(args.dpi),
                x0_frac=float(x0_frac),
                x1_frac=float(x1_frac),
                y0_frac=float(y0_frac),
                y1_frac=float(y1_frac),
                scales=list(args.scales),
                thr=float(args.thr),
                modes=wanted_modes,
                methods=wanted_methods,
                save_best=bool(args.save_best),
                save_rejected_best=bool(args.save_rejected_best),
                downscale=float(args.downscale),
            )
            rc2 = _run_one_backend(
                backend="pymupdf",
                pdf_path=pdf_path,
                template_path=template_path,
                out_dir=out_dir,
                page_index=int(page_index),
                dpi=int(args.dpi),
                x0_frac=float(x0_frac),
                x1_frac=float(x1_frac),
                y0_frac=float(y0_frac),
                y1_frac=float(y1_frac),
                scales=list(args.scales),
                thr=float(args.thr),
                modes=wanted_modes,
                methods=wanted_methods,
                save_best=bool(args.save_best),
                save_rejected_best=bool(args.save_rejected_best),
                downscale=float(args.downscale),
            )
            rc = rc or int(rc1 or rc2)
        return int(rc)

    rc = 0
    for page_index in pages_to_run:
        rc = rc or int(
            _run_one_backend(
                backend=str(back),  # type: ignore[arg-type]
                pdf_path=pdf_path,
                template_path=template_path,
                out_dir=out_dir,
                page_index=int(page_index),
                dpi=int(args.dpi),
                x0_frac=float(x0_frac),
                x1_frac=float(x1_frac),
                y0_frac=float(y0_frac),
                y1_frac=float(y1_frac),
                scales=list(args.scales),
                thr=float(args.thr),
                modes=wanted_modes,
                methods=wanted_methods,
                save_best=bool(args.save_best),
                save_rejected_best=bool(args.save_rejected_best),
                downscale=float(args.downscale),
            )
        )
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())


