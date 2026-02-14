#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
from PIL import Image, ImageOps

try:
    import pytesseract  # type: ignore
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: pytesseract\n"
        "Install it into your environment, e.g.:\n"
        "  .venv/bin/python -m pip install pytesseract\n"
        "Also ensure the 'tesseract' binary is installed on your system.\n"
    ) from e


_RE_ANY_DIGITS = re.compile(r"[0-9๐๑๒๓๔๕๖๗๘๙]+")
_RE_ARABIC_DIGITS = re.compile(r"[0-9]+")

# Common OCR confusions when the ground truth is digits.
_DIGIT_CONFUSION_TRANSLATION = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "Q": "0",
        "D": "0",  # occasionally seen in low-res scans
        "I": "1",
        "l": "1",
        "|": "1",
        "!": "1",
        "Z": "2",
        "z": "2",
        "S": "5",
        "s": "5",
        "G": "6",
        "B": "8",
    }
)


def _extract_arabic_digits_only(txt: str) -> str:
    """
    Extract the "best" Arabic-digit run (0-9) from OCR output.
    We apply a small confusion map first, then return the longest digit run.
    """
    norm = (txt or "").strip().translate(_DIGIT_CONFUSION_TRANSLATION)
    runs = _RE_ARABIC_DIGITS.findall(norm)
    if not runs:
        return ""
    # Prefer the longest run; if ties, keep the first.
    return max(runs, key=len)


@dataclass(frozen=True)
class OCRResult:
    path: str
    kind: str
    text: str
    number: str


def _infer_kind(p: Path) -> str:
    parts = {x.lower() for x in p.parts}
    if "district" in parts:
        return "district"
    if "partylist" in parts:
        return "partylist"
    return "unknown"


def _iter_inputs(input_dir: Path) -> Iterable[Path]:
    exts = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _render_pdf_page_to_pil(pdf_path: Path, *, page: int, dpi: int) -> Image.Image:
    doc = fitz.open(str(pdf_path))
    try:
        if page < 0 or page >= doc.page_count:
            raise ValueError(f"page out of range: {page} (pages: {doc.page_count})")
        pg = doc.load_page(page)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = pg.get_pixmap(matrix=mat, alpha=False)
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    finally:
        doc.close()


def _load_image(path: Path, *, page: int, dpi: int) -> Image.Image:
    if path.suffix.lower() == ".pdf":
        return _render_pdf_page_to_pil(path, page=page, dpi=dpi)
    return Image.open(path).convert("RGB")


def _preprocess(
    img: Image.Image,
    *,
    invert: bool,
    threshold: int | None,
    scale: float,
) -> Image.Image:
    if scale != 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
    g = ImageOps.grayscale(img)
    if invert:
        g = ImageOps.invert(g)
    if threshold is not None:
        g = g.point(lambda p: 255 if p > threshold else 0, mode="1").convert("L")
    return g


def _check_tesseract_available() -> None:
    try:
        subprocess.run(["tesseract", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "tesseract binary not found (or not runnable).\n"
            "Install it and ensure it's on PATH.\n"
            "- macOS (brew): brew install tesseract tesseract-lang\n"
            "- Ubuntu/Debian: sudo apt-get install -y tesseract-ocr tesseract-ocr-tha\n"
        ) from e


def _ocr_one(
    path: Path,
    *,
    mode: str,
    lang: str,
    psm: int,
    oem: int,
    whitelist: str | None,
    page: int,
    dpi: int,
    invert: bool,
    threshold: int | None,
    scale: float,
) -> OCRResult:
    img = _load_image(path, page=page, dpi=dpi)
    img = _preprocess(img, invert=invert, threshold=threshold, scale=scale)

    config_parts = [f"--psm {psm}", f"--oem {oem}"]
    if mode == "digits":
        # Tell Tesseract we expect only numbers and disable dictionary assists.
        # These settings typically reduce hallucinated characters in digit-only fields.
        config_parts.append("-c classify_bln_numeric_mode=1")
        config_parts.append("-c load_system_dawg=0")
        config_parts.append("-c load_freq_dawg=0")
    if whitelist:
        config_parts.append(f"-c tessedit_char_whitelist={whitelist}")
    config = " ".join(config_parts)

    txt = pytesseract.image_to_string(img, lang=lang, config=config)
    txt = (txt or "").strip()

    number = ""
    if mode == "digits":
        number = _extract_arabic_digits_only(txt)
    elif mode == "text":
        number = ""
    else:
        raise ValueError(f"unknown mode: {mode}")

    return OCRResult(path=str(path), kind=_infer_kind(path), text=txt, number=number)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="OCR cropped PDFs/images. Designed for Thai forms.")
    p.add_argument("--input-dir", required=True, help="Directory containing cropped PDFs/images (recursive).")
    p.add_argument("--out", required=True, help="Output JSONL path.")
    p.add_argument(
        "--mode",
        choices=["digits", "text"],
        default="text",
        help="digits = try to extract handwritten number; text = full OCR text. Default: text.",
    )
    p.add_argument(
        "--lang",
        default="tha+eng",
        help="Tesseract language(s). Default: tha+eng (requires Thai traineddata installed).",
    )
    p.add_argument("--page", type=int, default=0, help="0-based page index for PDFs. Default: 0.")
    p.add_argument("--dpi", type=int, default=220, help="Render DPI for PDFs. Default: 220.")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2), help="Concurrency. Default: ~half CPUs.")
    p.add_argument(
        "--psm",
        type=int,
        default=None,
        help=(
            "Tesseract page segmentation mode. "
            "If omitted: digits->7 (single line), text->6 (block of text)."
        ),
    )
    p.add_argument("--oem", type=int, default=1, help="Tesseract OCR engine mode. Default: 1 (LSTM).")
    p.add_argument("--invert", action="store_true", help="Invert colors before OCR.")
    p.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Optional binarization threshold (0-255). Example: 180. Default: none.",
    )
    p.add_argument("--scale", type=float, default=1.0, help="Resize factor before OCR. Example: 1.5. Default: 1.0.")
    p.add_argument(
        "--whitelist",
        default=None,
        help="Optional character whitelist. If omitted and mode=digits, a digits whitelist is used.",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for debugging/tuning (process only the first N files).",
    )
    args = p.parse_args(argv)

    _check_tesseract_available()

    input_dir = Path(args.input_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = list(_iter_inputs(input_dir))
    if not paths:
        print(f"No inputs found under: {input_dir}", file=sys.stderr)
        return 2
    if args.max_files is not None:
        paths = paths[: max(0, int(args.max_files))]

    whitelist = args.whitelist
    if whitelist is None and args.mode == "digits":
        # If you *know* the handwritten field is Arabic digits only, this reduces confusion a lot.
        # If you need Thai digits too, pass --whitelist "0123456789๐๑๒๓๔๕๖๗๘๙".
        whitelist = "0123456789"

    psm = int(args.psm) if args.psm is not None else (7 if args.mode == "digits" else 6)

    failures = 0
    with out_path.open("w", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = [
            ex.submit(
                _ocr_one,
                path,
                mode=str(args.mode),
                lang=str(args.lang),
                psm=psm,
                oem=int(args.oem),
                whitelist=str(whitelist) if whitelist else None,
                page=int(args.page),
                dpi=int(args.dpi),
                invert=bool(args.invert),
                threshold=int(args.threshold) if args.threshold is not None else None,
                scale=float(args.scale),
            )
            for path in paths
        ]

        for fut in as_completed(futs):
            try:
                res = fut.result()
            except Exception as e:
                failures += 1
                # Best-effort: write error row
                f.write(json.dumps({"error": str(e)}, ensure_ascii=False) + "\n")
                continue
            f.write(
                json.dumps(
                    {"path": res.path, "kind": res.kind, "text": res.text, "number": res.number},
                    ensure_ascii=False,
                )
                + "\n"
            )

    if failures:
        print(f"Done with {failures} failure(s). Output: {out_path}", file=sys.stderr)
        return 1

    print(f"Done. Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


