#!/usr/bin/env python3
"""Extract structured data from Thai election forms (สส.5/18) using Vision LLMs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

_fitz = None


def _get_fitz():
    """Lazy-import PyMuPDF so --help works without the dependency installed."""
    global _fitz
    if _fitz is None:
        try:
            import fitz  # type: ignore

            _fitz = fitz
        except ModuleNotFoundError as e:
            raise SystemExit(
                "Missing dependency: PyMuPDF\n"
                "Install it: pip install PyMuPDF\n"
            ) from e
    return _fitz

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are an expert at reading Thai election tally forms (แบบ สส.5/18).
Extract ALL data from this form image into structured JSON.

Rules:
1. Identify the form type from the header — e.g. "สส.5/18 แบ่งเขต" (constituency),
   "สส.5/18 บัญชีรายชื่อ" (party-list), or "สส.5/11".
2. Read header metadata: province (จังหวัด), constituency number (เขตเลือกตั้งที่),
   amphoe (อำเภอ/เขต), tambon (ตำบล/แขวง), unit number (หน่วยเลือกตั้งที่).
3. Read voter statistics from the dotted-line fields.
4. Read the candidate/party table: number, full name, party name, vote count (คะแนน).
5. Convert Thai numerals (๐๑๒๓๔๕๖๗๘๙) to Arabic (0123456789).
6. Use null for any field that is blank, illegible, or absent.
7. Return ONLY valid JSON — no markdown fences, no explanation text.

Return this exact JSON structure:
{
  "form_type": "สส.5/18 แบ่งเขต",
  "province": "string",
  "constituency": 1,
  "amphoe": "string",
  "tambon": "string",
  "unit_number": 1,
  "voter_stats": {
    "eligible_voters": null,
    "showed_up": null,
    "ballots_received": null,
    "ballots_used": null,
    "valid_ballots": null,
    "invalid_ballots": null,
    "no_vote_ballots": null,
    "total_candidate_votes": null
  },
  "candidates": [
    {"number": 1, "name": "string", "party": "string", "votes": 0}
  ]
}"""

# ---------------------------------------------------------------------------
# Helpers (patterns from run_typhoon_ocr.py)
# ---------------------------------------------------------------------------


def _get_tqdm():
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm
    except Exception:
        return None


def _fallback_progress(done: int, total: int, width: int = 28) -> str:
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    filled = int(round(width * (done / total)))
    bar = "=" * filled + "-" * (width - filled)
    return f"[{bar}] {done}/{total}"


def fingerprint(path: Path) -> str:
    stat = path.stat()
    h = hashlib.sha1()
    h.update(f"{stat.st_size}:{int(stat.st_mtime)}".encode("utf-8"))
    return h.hexdigest()


def render_page_png(pdf_path: Path, *, page_idx: int, dpi: int = 200) -> bytes:
    """Render a single PDF page to PNG bytes via PyMuPDF."""
    fitz = _get_fitz()
    doc = fitz.open(str(pdf_path))
    try:
        pg = doc.load_page(page_idx)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = pg.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def load_env_file(path: Path, *, override: bool = False) -> None:
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
            k, v = k.strip(), v.strip()
            if not k:
                continue
            if k in os.environ and not override:
                continue
            os.environ[k] = v
    except Exception:
        return


def load_done_set(manifest_jsonl: Path) -> set[tuple[str, int, str]]:
    """Set of (rel_path, page_num, fingerprint) already processed successfully."""
    done: set[tuple[str, int, str]] = set()
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
                    if r.get("status") == "ok":
                        done.add((r["rel_path"], int(r["page_num"]), r["fingerprint"]))
                except Exception:
                    continue
    except Exception:
        pass
    return done


def iter_pdfs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.pdf") if p.is_file())


# ---------------------------------------------------------------------------
# Gemini interaction
# ---------------------------------------------------------------------------


def _init_gemini(api_key: str, model_name: str):
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def extract_page(
    *,
    model,
    png_bytes: bytes,
    retries: int,
    retry_base_sleep: float,
) -> dict[str, Any]:
    """Send one page image to Gemini and return parsed JSON."""
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = model.generate_content(
                [EXTRACTION_PROMPT, {"mime_type": "image/png", "data": png_bytes}],
                generation_config={"temperature": 0.0, "response_mime_type": "application/json"},
            )
            text = resp.text.strip()
            # Fallback: strip markdown fences if model ignores mime_type constraint
            if text.startswith("```"):
                text = text[text.index("\n") + 1 :]
            if text.endswith("```"):
                text = text[:-3].rstrip()
            return json.loads(text)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(retry_base_sleep * (2**attempt) + random.random())
    raise last_err  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Ollama interaction (OpenAI-compatible API)
# ---------------------------------------------------------------------------


def extract_page_ollama(
    *,
    ollama_url: str,
    model_name: str,
    png_bytes: bytes,
    retries: int,
    retry_base_sleep: float,
) -> dict[str, Any]:
    """Send one page image to a local Ollama vision model and return parsed JSON."""
    import base64

    import requests  # available from google-generativeai deps

    b64_img = base64.b64encode(png_bytes).decode("ascii")
    # /no_think disables Qwen3's chain-of-thought mode which puts output in
    # the 'reasoning' field and leaves 'content' empty.
    prompt_text = "/no_think\n" + EXTRACTION_PROMPT
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a data extraction assistant. Respond with valid JSON only.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                ],
            },
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{ollama_url}/v1/chat/completions",
                json=payload,
                timeout=600,
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            # Qwen3 thinking mode: content may be empty, output in 'reasoning'
            text = (msg.get("content") or msg.get("reasoning") or "").strip()
            if text.startswith("```"):
                text = text[text.index("\n") + 1 :]
            if text.endswith("```"):
                text = text[:-3].rstrip()
            return json.loads(text)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(retry_base_sleep * (2**attempt) + random.random())
    raise last_err  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Work-item builders
# ---------------------------------------------------------------------------


def _try_parse_dirpath(rel: str) -> dict[str, Any]:
    """Best-effort directory metadata extraction. Returns empty dict on failure."""
    try:
        from parse_dirpath import parse_election_path
        return parse_election_path(rel)
    except Exception:
        return {}


def build_work_items(
    input_path: Path,
    done_set: set[tuple[str, int, str]],
) -> list[dict[str, Any]]:
    """Build page-level work items. Only odd pages (1-based: 1,3,5,...) are vote tables."""
    if input_path.is_file():
        pdfs = [input_path]
        base = input_path.parent
    else:
        pdfs = iter_pdfs(input_path)
        base = input_path

    items: list[dict[str, Any]] = []
    for pdf in pdfs:
        rel = str(pdf.relative_to(base))
        fp = fingerprint(pdf)
        fitz = _get_fitz()
        doc = fitz.open(str(pdf))
        n_pages = doc.page_count
        doc.close()

        # Parse directory-derived metadata once per PDF
        dir_meta = _try_parse_dirpath(rel)

        for idx in range(0, n_pages, 2):  # 0-indexed → odd pages (1,3,5,...)
            page_num = idx + 1
            if (rel, page_num, fp) in done_set:
                continue
            item: dict[str, Any] = {
                "pdf_path": pdf,
                "rel_path": rel,
                "page_idx": idx,
                "page_num": page_num,
                "fingerprint": fp,
                "n_pages": n_pages,
            }
            if dir_meta:
                item["dir_meta"] = dir_meta
            items.append(item)
    return items


def process_one(
    *,
    item: dict[str, Any],
    backend: str,
    model_name: str,
    dpi: int,
    retries: int,
    retry_base_sleep: float,
    gemini_model=None,
    ollama_url: str = "",
    rate_limiter: Optional[threading.Semaphore] = None,
) -> dict[str, Any]:
    """Render one page + call Vision LLM → manifest record."""
    t0 = time.time()
    try:
        png_bytes = render_page_png(item["pdf_path"], page_idx=item["page_idx"], dpi=dpi)
        if rate_limiter:
            rate_limiter.acquire()
        if backend == "ollama":
            extraction = extract_page_ollama(
                ollama_url=ollama_url,
                model_name=model_name,
                png_bytes=png_bytes,
                retries=retries,
                retry_base_sleep=retry_base_sleep,
            )
        else:
            extraction = extract_page(
                model=gemini_model,
                png_bytes=png_bytes,
                retries=retries,
                retry_base_sleep=retry_base_sleep,
            )
        result = {
            "rel_path": item["rel_path"],
            "page_num": item["page_num"],
            "fingerprint": item["fingerprint"],
            "status": "ok",
            "extraction": extraction,
            "model": model_name,
            "seconds": round(time.time() - t0, 3),
        }
        if "dir_meta" in item:
            result["dir_meta"] = item["dir_meta"]
        return result
    except Exception as e:
        return {
            "rel_path": item["rel_path"],
            "page_num": item["page_num"],
            "fingerprint": item["fingerprint"],
            "status": "error",
            "error": str(e),
            "model": model_name,
            "seconds": round(time.time() - t0, 3),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Extract structured data from Thai election forms (สส.5/18) via Vision LLM.",
        allow_abbrev=False,
    )
    ap.add_argument("--input", required=True, help="PDF file or directory of PDFs.")
    ap.add_argument("--out-root", required=True, help="Output directory (manifest lives here).")
    ap.add_argument("--manifest-jsonl", required=True, help="JSONL manifest path (append-only, for resume).")
    ap.add_argument("--backend", choices=["gemini", "ollama"], default="gemini", help="Vision LLM backend. Default: gemini.")
    ap.add_argument("--model", default=None, help="Model name. Default: gemini-2.0-flash / qwen3-vl:8b.")
    ap.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama API URL. Default: http://localhost:11434.")
    ap.add_argument("--workers", type=int, default=None, help="Concurrent workers. Default: 5 (gemini) / 1 (ollama).")
    ap.add_argument("--max-files", type=int, default=0, help="Max PDFs to process (0 = all).")
    ap.add_argument("--max-pages", type=int, default=0, help="Max pages to process (0 = all).")
    ap.add_argument("--dpi", type=int, default=200, help="Render DPI for PDFs. Default: 200.")
    ap.add_argument("--retries", type=int, default=3, help="Retries per page. Default: 3.")
    ap.add_argument("--retry-sleep", type=float, default=2.0, help="Base backoff sleep (s). Default: 2.0.")
    ap.add_argument("--dry-run", action="store_true", help="Scan inputs only, no API calls.")
    ap.add_argument("--env-file", default="env.local", help="Env file path. Default: env.local.")
    args = ap.parse_args(argv)

    # --- defaults per backend ---
    if args.model is None:
        args.model = "qwen3-vl:8b" if args.backend == "ollama" else "gemini-2.0-flash"
    if args.workers is None:
        args.workers = 1 if args.backend == "ollama" else 5

    # --- env & API key ---
    load_env_file(Path(args.env_file))
    api_key = ""
    if args.backend == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key or api_key in {"YOUR_KEY_HERE", "CHANGE_ME"}:
            if args.dry_run:
                print("Warning: GEMINI_API_KEY not set. Dry-run will still scan inputs.")
            else:
                raise SystemExit("Missing env var: GEMINI_API_KEY (set in env.local or environment)")

    # --- paths ---
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input does not exist: {input_path}")
    out_root = Path(args.out_root)
    manifest_jsonl = Path(args.manifest_jsonl)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # --- build work items ---
    done_set = load_done_set(manifest_jsonl)
    items = build_work_items(input_path, done_set)

    if args.max_files and args.max_files > 0:
        seen: list[Path] = []
        filtered: list[dict[str, Any]] = []
        for item in items:
            if item["pdf_path"] not in seen:
                if len(seen) >= args.max_files:
                    break
                seen.append(item["pdf_path"])
            filtered.append(item)
        items = filtered

    if args.max_pages and args.max_pages > 0:
        items = items[: args.max_pages]

    skipped = len(done_set)

    # --- dry-run ---
    if args.dry_run:
        print("Gemini extraction dry-run (no API calls).")
        print(f"  input:    {input_path}")
        print(f"  out_root: {out_root}")
        print(f"  manifest: {manifest_jsonl}")
        print(f"  model:    {args.model}")
        print(f"  pending pages: {len(items)}  already done: {skipped}")
        if items:
            for item in items[:8]:
                print(f"  - {item['rel_path']} page {item['page_num']}")
            if len(items) > 8:
                print(f"  ... and {len(items) - 8} more")
        return 0

    if not items:
        print("Nothing to process (all pages already in manifest).")
        return 0

    # --- init backend ---
    gemini_model = None
    if args.backend == "gemini":
        gemini_model = _init_gemini(api_key, args.model)

    # --- rate limiter (replenishes tokens via a background timer) ---
    rate_limiter: Optional[threading.Semaphore] = None
    rate_timer: Optional[threading.Timer] = None
    if args.backend == "gemini" and args.workers > 1:
        # Gemini paid tier: 1000 RPM; free tier: 15 RPM.
        # Use a semaphore that releases one token per 0.1s (≈600 RPM safe default).
        rate_limiter = threading.Semaphore(args.workers)
        _stop_rate = threading.Event()

        def _refill():
            while not _stop_rate.is_set():
                time.sleep(0.1)
                try:
                    rate_limiter.release()
                except ValueError:
                    pass

        rate_timer = threading.Thread(target=_refill, daemon=True)
        rate_timer.start()

    # --- process ---
    manifest_lock = threading.Lock()
    processed = ok_count = error_count = 0
    total = len(items)
    start = time.time()

    desc = f"{args.backend}:{args.model}"
    tqdm_cls = _get_tqdm()
    pbar = tqdm_cls(total=total, desc=desc, unit="page") if tqdm_cls else None
    last_fb = 0.0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {
            ex.submit(
                process_one,
                item=item,
                backend=args.backend,
                model_name=args.model,
                dpi=args.dpi,
                retries=args.retries,
                retry_base_sleep=args.retry_sleep,
                gemini_model=gemini_model,
                ollama_url=args.ollama_url,
                rate_limiter=rate_limiter,
            ): item
            for item in items
        }
        try:
            for fut in as_completed(futs):
                r = fut.result()
                processed += 1
                if r["status"] == "ok":
                    ok_count += 1
                else:
                    error_count += 1
                    print(f"\n  ERROR {r['rel_path']} p{r['page_num']}: {r.get('error', '?')}")

                with manifest_lock:
                    with manifest_jsonl.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

                if pbar:
                    pbar.update(1)
                else:
                    now = time.time()
                    if processed == total or (now - last_fb) >= 0.25:
                        print("\r" + _fallback_progress(processed, total), end="", flush=True)
                        last_fb = now
        finally:
            if pbar:
                pbar.close()
            else:
                print()

    # Stop rate limiter background thread
    if rate_limiter is not None:
        _stop_rate.set()

    elapsed = round(time.time() - start, 1)
    print(
        f"Done. processed={processed} ok={ok_count} error={error_count} "
        f"skipped={skipped} ({elapsed}s)"
    )
    print(f"Manifest: {manifest_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
