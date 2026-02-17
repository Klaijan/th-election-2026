#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import fitz  # PyMuPDF

def _get_tqdm():
    """
    Optional progress bar dependency.
    If tqdm isn't installed, return None and we fall back to a simple text progress indicator.
    """
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm
    except Exception:
        return None


def _fallback_progress_str(done: int, total: int, width: int = 28) -> str:
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    filled = int(round(width * (done / total)))
    bar = "=" * filled + "-" * (width - filled)
    return f"[{bar}] {done}/{total}"


def _get_ocr_document():
    """
    Import Typhoon OCR lazily so `--help` works even when the dependency isn't installed.
    """
    try:
        from typhoon_ocr import ocr_document  # type: ignore

        return ocr_document
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: typhoon_ocr\n"
            "Install it into your environment (package name depends on your setup), e.g.:\n"
            "  .venv/bin/python -m pip install typhoon-ocr\n"
            "Then re-run.\n"
        ) from e


def fingerprint(path: Path) -> str:
    """
    Stable-ish fingerprint to support skipping previously processed files.
    Uses file size + mtime (fast; good enough for batch pipelines).
    """
    stat = path.stat()
    h = hashlib.sha1()
    h.update(f"{stat.st_size}:{int(stat.st_mtime)}".encode("utf-8"))
    return h.hexdigest()


def num_pages(path: Path) -> int:
    """
    Page count for PDFs; images return 1.
    """
    if path.suffix.lower() != ".pdf":
        return 1
    doc = fitz.open(str(path))
    try:
        return int(doc.page_count) or 1
    finally:
        doc.close()


def _render_pdf_page_to_tmp_png(pdf_path: Path, *, page_num_1based: int, dpi: int) -> Path:
    """
    Render a PDF page to a temporary PNG using PyMuPDF.
    This avoids relying on external PDF utilities (e.g. Poppler) inside the Typhoon client.
    """
    if page_num_1based < 1:
        raise ValueError(f"page_num must be >= 1 (got {page_num_1based})")

    doc = fitz.open(str(pdf_path))
    try:
        idx = int(page_num_1based) - 1
        if idx < 0 or idx >= doc.page_count:
            raise ValueError(f"page out of range: {page_num_1based} (pages: {doc.page_count})")
        pg = doc.load_page(idx)
        mat = fitz.Matrix(float(dpi) / 72.0, float(dpi) / 72.0)
        pix = pg.get_pixmap(matrix=mat, alpha=False)
        b = pix.tobytes("png")
    finally:
        doc.close()

    fd, tmp_path = tempfile.mkstemp(prefix="typhoon_ocr_", suffix=".png")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(b)
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise
    return Path(tmp_path)


def read_existing_fingerprint(out_md: Path) -> Optional[str]:
    """
    Reads fingerprint embedded in the first lines of the output markdown.
    We write a header like:
      <!-- rel_path=... -->
      <!-- fingerprint=... -->
    """
    if not out_md.exists():
        return None
    try:
        with out_md.open("r", encoding="utf-8") as f:
            for _ in range(12):
                line = f.readline()
                if not line:
                    break
                s = line.strip()
                if s.startswith("<!-- fingerprint=") and s.endswith("-->"):
                    return s[len("<!-- fingerprint=") : -len("-->")].strip()
    except Exception:
        return None
    return None


def iter_inputs(root: Path) -> list[Path]:
    exts = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in exts]


def ocr_one(
    *,
    path: Path,
    rel: str,
    fp: str,
    out_md: Path,
    base_url: str,
    api_key: str,
    model: str,
    retries: int,
    retry_base_sleep: float,
    ocr_document_func: Any,
    dpi: int,
) -> dict[str, Any]:
    t0 = time.time()
    try:
        n_pages = num_pages(path)
        md_parts: list[str] = []

        # Header for robust skipping even without manifest.
        md_parts.append(f"<!-- rel_path={rel} -->")
        md_parts.append(f"<!-- fingerprint={fp} -->")
        md_parts.append(f"<!-- model={model} -->")
        md_parts.append("")

        for page in range(1, n_pages + 1):  # 1-based pages
            last_err: Optional[Exception] = None
            tmp_img: Optional[Path] = None
            ocr_path = str(path)
            page_num = page
            if path.suffix.lower() == ".pdf":
                tmp_img = _render_pdf_page_to_tmp_png(path, page_num_1based=page, dpi=int(dpi))
                ocr_path = str(tmp_img)
                page_num = 1  # images always use page_num=1
            for attempt in range(retries + 1):
                try:
                    md = ocr_document_func(
                        pdf_or_image_path=ocr_path,
                        page_num=page_num,
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                    )
                    md_parts.append(f"\n\n<!-- PAGE {page}/{n_pages} -->\n\n{md}")
                    last_err = None
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    last_err = e
                    sleep_s = retry_base_sleep * (2**attempt) + random.random()
                    time.sleep(sleep_s)
            if last_err is not None:
                if tmp_img is not None:
                    try:
                        tmp_img.unlink()
                    except Exception:
                        pass
                raise last_err
            if tmp_img is not None:
                try:
                    tmp_img.unlink()
                except Exception:
                    pass

        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(md_parts), encoding="utf-8")

        return {
            "rel_path": rel,
            "fingerprint": fp,
            "status": "ok",
            "in_path": str(path),
            "out_md": str(out_md),
            "n_pages": n_pages,
            "seconds": round(time.time() - t0, 3),
        }
    except Exception as e:
        return {
            "rel_path": rel,
            "fingerprint": fp,
            "status": "error",
            "in_path": str(path),
            "out_md": str(out_md),
            "error": str(e),
            "seconds": round(time.time() - t0, 3),
        }


def load_done_map(manifest_jsonl: Path) -> dict[tuple[str, str], str]:
    """
    Load previous JSONL manifest into a map keyed by (rel_path, fingerprint) -> status.
    Used to skip without parquet.
    """
    done: dict[tuple[str, str], str] = {}
    if not manifest_jsonl.exists():
        return done
    try:
        with manifest_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    rel = r.get("rel_path", "")
                    fp = r.get("fingerprint", "")
                    st = r.get("status", "")
                    if rel and fp:
                        done[(rel, fp)] = st
                except Exception:
                    continue
    except Exception:
        return done
    return done


def load_env_file(path: Path, *, override: bool = False) -> None:
    """
    Load KEY=VALUE lines into os.environ.
    - Ignores blank lines and comments starting with '#'
    - Does not require quoting
    - If override=False, keeps existing env vars
    """
    if not path.exists() or not path.is_file():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not k:
                continue
            if (k in os.environ) and not override:
                continue
            os.environ[k] = v
    except Exception:
        # If env parsing fails, just fall back to existing env vars.
        return


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run Typhoon OCR over PDFs/images and write Markdown + manifests (resume-friendly).",
        allow_abbrev=False,
    )
    # New flag name (preferred) + legacy alias (matches older scripts / muscle memory)
    ap.add_argument("--raw-root", default=None, help="Root directory to scan for PDFs/images (recursive).")
    ap.add_argument("--input-root", dest="raw_root", default=None, help="Alias for --raw-root.")
    ap.add_argument("--out-root", required=True, help="Root directory to write .md outputs (mirrors raw-root tree).")
    ap.add_argument("--manifest-jsonl", default=None, help="Path to JSONL manifest file (append-only).")
    ap.add_argument("--manifest", dest="manifest_jsonl", default=None, help="Alias for --manifest-jsonl.")
    ap.add_argument(
        "--stats-jsonl",
        default="",
        help=(
            "Optional JSONL file to append per-run summary stats. "
            "Default: <manifest-jsonl>.stats.jsonl"
        ),
    )
    ap.add_argument(
        "--no-stats",
        action="store_true",
        help="Disable writing per-run summary stats JSONL file.",
    )
    ap.add_argument("--workers", type=int, default=3, help="Thread workers (API calls). Default: 3.")
    ap.add_argument("--max-seconds", type=int, default=7200, help="Stop after this many seconds (chunking). Default: 7200.")
    ap.add_argument("--max-files", type=int, default=0, help="0 = no limit. Otherwise process only first N pending.")
    ap.add_argument("--model", default=os.environ.get("TYPHOON_MODEL", "typhoon-ocr"), help="Model name.")
    ap.add_argument("--dpi", type=int, default=220, help="Render DPI for PDFs (PDF->PNG). Default: 220.")
    ap.add_argument("--retries", type=int, default=3, help="Retries per page. Default: 3.")
    ap.add_argument("--retry-sleep", type=float, default=2.0, help="Base backoff sleep seconds. Default: 2.0.")
    ap.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a progress bar while processing files. Default: true.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Preflight only: scan inputs, compute pending/skipped, print summary, then exit without calling the API.",
    )
    ap.add_argument(
        "--env-file",
        default="env.local",
        help="Env file to load before running (KEY=VALUE). Default: env.local",
    )
    ap.add_argument(
        "--env-override",
        action="store_true",
        help="If set, values in --env-file override already-set environment variables.",
    )
    args = ap.parse_args(argv)

    # Load env.local (or the provided file) by default so users don't need to `source` it.
    load_env_file(Path(str(args.env_file)), override=bool(args.env_override))

    # Defaults / compatibility:
    # - docs often use TYPHOON_OCR_API_KEY; keep supporting repo's TYPHOON_API_KEY too
    # - default base URL if not provided
    base_url = (os.environ.get("TYPHOON_BASE_URL") or "https://api.opentyphoon.ai/v1").strip()
    api_key = (os.environ.get("TYPHOON_API_KEY") or os.environ.get("TYPHOON_OCR_API_KEY") or "").strip()
    if not api_key or api_key in {"YOUR_KEY_HERE", "CHANGE_ME"}:
        if args.dry_run:
            print(
                "Warning: missing/placeholder Typhoon API key "
                "(set TYPHOON_API_KEY or TYPHOON_OCR_API_KEY; see env.example)."
            )
            print("Dry-run will still scan inputs and compute pending/skipped, but real OCR will fail until you set a real key.")
        else:
            raise SystemExit("Missing env var: TYPHOON_API_KEY (or compatible alias: TYPHOON_OCR_API_KEY)")

    if not args.raw_root:
        raise SystemExit("Missing required flag: --raw-root (or legacy alias: --input-root)")
    if not args.manifest_jsonl:
        raise SystemExit("Missing required flag: --manifest-jsonl (or legacy alias: --manifest)")

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    manifest_jsonl = Path(args.manifest_jsonl)
    stats_jsonl = None
    if not bool(args.no_stats):
        stats_jsonl_str = str(args.stats_jsonl).strip()
        if not stats_jsonl_str:
            # Default: alongside the manifest, replacing ".jsonl" with ".stats.jsonl" when applicable.
            if manifest_jsonl.suffix.lower() == ".jsonl":
                stats_jsonl = manifest_jsonl.with_suffix(".stats.jsonl")
            else:
                stats_jsonl = Path(str(manifest_jsonl) + ".stats.jsonl")
        else:
            stats_jsonl = Path(stats_jsonl_str)

    if not raw_root.exists():
        raise SystemExit(f"Raw root does not exist: {raw_root}")

    out_root.mkdir(parents=True, exist_ok=True)
    manifest_jsonl.parent.mkdir(parents=True, exist_ok=True)

    inputs = iter_inputs(raw_root)
    if not inputs:
        raise SystemExit(f"No PDFs/images found under: {raw_root}")

    start = time.time()
    pending: list[tuple[Path, str, str, Path]] = []
    skipped_rows: list[dict[str, Any]] = []

    done_map = load_done_map(manifest_jsonl)

    for path in inputs:
        rel = str(path.relative_to(raw_root))
        fp = fingerprint(path)
        out_md = (out_root / rel).with_suffix(".md")

        existing_fp = read_existing_fingerprint(out_md)
        if existing_fp == fp:
            skipped_rows.append({"rel_path": rel, "fingerprint": fp, "status": "skipped", "out_md": str(out_md)})
            continue

        if (rel, fp) in done_map and out_md.exists():
            skipped_rows.append({"rel_path": rel, "fingerprint": fp, "status": "skipped", "out_md": str(out_md)})
            continue

        pending.append((path, rel, fp, out_md))

    # Write skipped rows immediately so manifest isn't empty while running.
    if skipped_rows:
        with manifest_jsonl.open("a", encoding="utf-8") as f:
            for r in skipped_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.max_files and args.max_files > 0:
        pending = pending[: int(args.max_files)]

    if args.dry_run:
        print("Typhoon OCR dry-run preflight (no API calls).")
        print(f"raw_root: {raw_root}")
        print(f"out_root: {out_root}")
        print(f"manifest_jsonl: {manifest_jsonl}")
        print(f"inputs: {len(inputs)}  pending: {len(pending)}  skipped: {len(skipped_rows)}")
        if pending:
            preview_n = min(8, len(pending))
            print("Pending preview:")
            for (path, rel, fp, out_md) in pending[:preview_n]:
                _ = fp  # avoid unused in preview
                print(f"- {rel} -> {out_md}")
            if len(pending) > preview_n:
                print(f"... and {len(pending) - preview_n} more pending files")
        return 0

    if not pending:
        print("Nothing to OCR (all outputs already exist with matching fingerprint).")
        return 0

    # Import Typhoon OCR client only when we're actually about to call the API.
    ocr_document_func = _get_ocr_document()

    processed = 0
    ok_count = 0
    error_count = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = []
        for (path, rel, fp, out_md) in pending:
            futs.append(
                ex.submit(
                    ocr_one,
                    path=path,
                    rel=rel,
                    fp=fp,
                    out_md=out_md,
                    base_url=str(base_url),
                    api_key=str(api_key),
                    model=str(args.model),
                    retries=int(args.retries),
                    retry_base_sleep=float(args.retry_sleep),
                    ocr_document_func=ocr_document_func,
                    dpi=int(args.dpi),
                )
            )

        total = len(futs)
        tqdm_mod = _get_tqdm() if bool(args.progress) else None
        pbar = tqdm_mod(total=total, desc="Typhoon OCR", unit="file") if tqdm_mod else None
        last_fallback_print = 0.0

        stop_early = False
        try:
            for fut in as_completed(futs):
                r = fut.result()
                processed += 1
                st = str(r.get("status", ""))
                if st == "ok":
                    ok_count += 1
                elif st == "error":
                    error_count += 1
                with manifest_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

                if pbar is not None:
                    pbar.update(1)
                elif bool(args.progress):
                    # Simple fallback: render a basic bar, rate-limited to avoid spam.
                    now = time.time()
                    if processed == total or (now - last_fallback_print) >= 0.25:
                        print("\r" + _fallback_progress_str(processed, total), end="", flush=True)
                        last_fallback_print = now

                if (time.time() - start) >= int(args.max_seconds):
                    print(f"\nReached --max-seconds={args.max_seconds}. Stopping early.")
                    stop_early = True
                    break
        finally:
            if pbar is not None:
                pbar.close()
                # Ensure the final tqdm refresh is written before we print normal lines below.
                try:
                    import sys as _sys

                    _sys.stdout.flush()
                except Exception:
                    pass
            elif bool(args.progress):
                # End the \r line.
                print("")

        if stop_early:
            # Cancel remaining futures (Python 3.9+ supports cancel_futures).
            for f in futs:
                f.cancel()

    # Persist a per-run summary into a separate stats JSONL (for tracking success/failure rates over time).
    run_summary = {
        "row_type": "run_summary",
        "ts": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw_root),
        "out_root": str(out_root),
        "manifest_jsonl": str(manifest_jsonl),
        "model": str(args.model),
        "workers": int(args.workers),
        "max_seconds": int(args.max_seconds),
        "max_files": int(args.max_files),
        "inputs_total": len(inputs),
        "skipped_preexisting": len(skipped_rows),
        "pending_total": len(pending),
        "processed": processed,
        "ok": ok_count,
        "error": error_count,
        "stop_early": bool(stop_early),
        "seconds": round(time.time() - start, 3),
    }
    if stats_jsonl is not None:
        stats_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with stats_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(run_summary, ensure_ascii=False) + "\n")

    def _log_line(s: str) -> None:
        # If tqdm is active, use tqdm.write() to avoid corrupting the progress bar output.
        if bool(args.progress) and (tqdm_mod is not None):
            try:
                tqdm_mod.write(s)
                return
            except Exception:
                pass
        print(s)

    _log_line(
        "Done chunk. "
        f"processed={processed} ok={ok_count} error={error_count} skipped={len(skipped_rows)} "
        f"(workers={args.workers}, model={args.model})"
    )
    _log_line(f"Manifest JSONL: {manifest_jsonl}")
    if stats_jsonl is not None:
        _log_line(f"Stats JSONL:    {stats_jsonl}")
    _log_line(f"Outputs:  {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


