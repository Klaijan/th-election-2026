#!/usr/bin/env python3
from __future__ import annotations

import argparse

import ocr_cropped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Convenience wrapper around ocr_cropped.py for the repo's sample data.\n"
            "Defaults to: --input-dir data/sample/cropped --lang tha+eng --out data/sample/ocr_auto.jsonl"
        )
    )
    p.add_argument("--input-dir", default="data/sample/cropped", help="Default: data/sample/cropped")
    p.add_argument("--out", default="data/sample/ocr_auto.jsonl", help="Default: data/sample/ocr_auto.jsonl")
    p.add_argument("--mode", choices=["digits", "text"], default="text", help="Default: text")
    p.add_argument("--lang", default="tha+eng", help="Default: tha+eng")
    p.add_argument("--page", type=int, default=0, help="0-based page index for PDFs. Default: 0")
    p.add_argument("--dpi", type=int, default=220, help="Render DPI for PDFs. Default: 220")
    p.add_argument("--workers", type=int, default=None, help="Concurrency. Default: use ocr_cropped.py default")
    p.add_argument("--psm", type=int, default=None, help="Tesseract PSM. Default: use ocr_cropped.py default")
    p.add_argument("--oem", type=int, default=1, help="Tesseract OEM. Default: 1 (LSTM)")
    p.add_argument("--invert", action="store_true", help="Invert colors before OCR")
    p.add_argument("--threshold", type=int, default=None, help="Optional binarization threshold (0-255)")
    p.add_argument("--scale", type=float, default=1.0, help="Resize factor before OCR. Default: 1.0")
    p.add_argument("--whitelist", default=None, help="Optional character whitelist")
    p.add_argument("--max-files", type=int, default=None, help="Optional limit for debugging/tuning")
    args = p.parse_args(argv)

    forwarded: list[str] = [
        "--input-dir",
        str(args.input_dir),
        "--out",
        str(args.out),
        "--mode",
        str(args.mode),
        "--lang",
        str(args.lang),
        "--page",
        str(int(args.page)),
        "--dpi",
        str(int(args.dpi)),
        "--oem",
        str(int(args.oem)),
        "--scale",
        str(float(args.scale)),
    ]
    if args.workers is not None:
        forwarded += ["--workers", str(int(args.workers))]
    if args.psm is not None:
        forwarded += ["--psm", str(int(args.psm))]
    if args.invert:
        forwarded += ["--invert"]
    if args.threshold is not None:
        forwarded += ["--threshold", str(int(args.threshold))]
    if args.whitelist:
        forwarded += ["--whitelist", str(args.whitelist)]
    if args.max_files is not None:
        forwarded += ["--max-files", str(int(args.max_files))]

    return int(ocr_cropped.main(forwarded))


if __name__ == "__main__":
    raise SystemExit(main())


