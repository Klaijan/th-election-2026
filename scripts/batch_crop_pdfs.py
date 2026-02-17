#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _infer_kind(pdf_path: Path) -> str | None:
    """
    Infer 'district' or 'partylist' from any directory component in the path.
    """
    parts = {p.lower() for p in pdf_path.parts}
    if "district" in parts:
        return "district"
    if "partylist" in parts:
        return "partylist"
    return None


def _iter_pdfs(input_dir: Path) -> list[Path]:
    return [p for p in sorted(input_dir.rglob("*.pdf")) if p.is_file()]


def _run_one(
    *,
    py_exe: str,
    crop_script: Path,
    pdf_path: Path,
    out_pdf: Path,
    page: int,
    dpi: int,
) -> tuple[Path, int, str]:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        py_exe,
        str(crop_script),
        "--pdf",
        str(pdf_path),
        "--out",
        str(out_pdf),
        "--page",
        str(page),
        "--dpi",
        str(dpi),
        # Note: we rely on crop_pdf_page.py defaults for the crop rect unless user passes rect in that script.
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return pdf_path, proc.returncode, proc.stdout


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Batch crop PDFs by calling crop_pdf_page.py.\n"
            "Outputs are written under <out-root>/cropped/<district|partylist>/ with the same filename."
        )
    )
    p.add_argument("--input-dir", required=True, help="Directory to search for PDFs (recursive).")
    p.add_argument(
        "--out-root",
        required=True,
        help="Root directory that will contain the 'cropped/' folder (e.g. data/sample).",
    )
    p.add_argument(
        "--crop-script",
        default="crop_pdf_page.py",
        help="Path to crop script. Default: crop_pdf_page.py (in current working directory).",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use when invoking the crop script. Default: current interpreter.",
    )
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2), help="Concurrency. Default: ~half CPUs.")
    p.add_argument("--page", type=int, default=0, help="0-based page index to crop. Default: 0.")
    p.add_argument("--dpi", type=int, default=200, help="Render DPI (used for image output; harmless for PDF output). Default: 200.")
    p.add_argument(
        "--on-unknown",
        choices=["skip", "error", "unknown"],
        default="error",
        help="What to do if kind cannot be inferred. Default: error.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = p.parse_args(argv)

    input_dir = Path(args.input_dir)
    out_root = Path(args.out_root)
    crop_script = Path(args.crop_script)

    if not input_dir.exists():
        print(f"Input dir does not exist: {input_dir}", file=sys.stderr)
        return 2
    if not crop_script.exists():
        print(f"Crop script not found: {crop_script}", file=sys.stderr)
        return 3

    pdfs = _iter_pdfs(input_dir)
    if not pdfs:
        print(f"No PDFs found under: {input_dir}", file=sys.stderr)
        return 4

    failures: list[tuple[Path, str]] = []

    with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = []
        for pdf in pdfs:
            kind = _infer_kind(pdf)
            if kind is None:
                if args.on_unknown == "skip":
                    continue
                if args.on_unknown == "unknown":
                    kind = "unknown"
                else:
                    failures.append((pdf, "cannot infer kind (district/partylist) from path"))
                    continue

            out_pdf = out_root / "cropped" / kind / pdf.name
            if out_pdf.exists() and not args.overwrite:
                continue

            futs.append(
                ex.submit(
                    _run_one,
                    py_exe=str(args.python),
                    crop_script=crop_script,
                    pdf_path=pdf,
                    out_pdf=out_pdf,
                    page=int(args.page),
                    dpi=int(args.dpi),
                )
            )

        for fut in as_completed(futs):
            pdf_path, code, out = fut.result()
            if code != 0:
                failures.append((pdf_path, out.strip() or f"exit code {code}"))

    if failures:
        print(f"Completed with {len(failures)} failure(s):", file=sys.stderr)
        for pdf, msg in failures[:50]:
            print(f"- {pdf}: {msg}", file=sys.stderr)
        if len(failures) > 50:
            print(f"... and {len(failures) - 50} more", file=sys.stderr)
        return 1

    print(f"Done. Cropped {len(pdfs)} PDFs into: {out_root / 'cropped'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


