#!/usr/bin/env python3
"""Parse Thai election directory paths to extract structured metadata.

Handles the highly inconsistent naming conventions across provinces in data/raw/.

Usage:
    # Scan all PDFs under a directory, output CSV
    python scripts/parse_dirpath.py data/raw/ --csv data/parsed_paths.csv

    # Single path
    python scripts/parse_dirpath.py "data/raw/ลำปาง/เขตเลือกตั้งที่ 3/อำเภอแม่ทะ/1.ตำบลบ้านบอม/1.แบบแบ่งเขต 5-18/หน่วย3 ม.3/บอม 5-18 น3ม3.pdf"
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Regex patterns for each metadata field
# ---------------------------------------------------------------------------

RE_CONSTITUENCY = re.compile(r"เขตเลือกตั้งที่\s*(\d+)")

# Amphoe: "อำเภอX", "อ.X" (with possible leading number),
# also "เทศบาลนครX", "เทศบาลเมืองX" (city/town municipality as amphoe-level)
RE_AMPHOE = re.compile(
    r"(?:\d+\.)?\s*(?:อำเภอ|อ\.|เทศบาลนคร|เทศบาลเมือง)\s*(.+?)(?:\s+5[-/]?1[78].*)?$"
)

# Tambon/municipality: "ตำบลX", "ต.X", "ทม.X" (เทศบาลเมือง), "ทต.X" (เทศบาลตำบล), "อบต.X"
RE_TAMBON = re.compile(
    r"(?:\d+\.)?\s*(?:ตำบล|ต\.|ทม\.|ทต\.|อบต\.)\s*(.+?)(?:\.pdf)?(?:\s+รวม)?$",
    re.IGNORECASE,
)

# Tambon from filename: bare name before "ชุดที่" e.g. "พระบาท ชุดที่56.pdf"
RE_TAMBON_FROM_FILENAME = re.compile(
    r"^([ก-๙]+(?:[ก-๙\s]*[ก-๙]+)?)\s+ชุดที่",
)

# Form type: district (แบ่งเขต) vs party-list (บัญชีรายชื่อ / บช)
# District patterns: "แบ่งเขต", "ส.ส.5-18" (without บช), "สส5-18" (without บช)
RE_FORM_PARTYLIST = re.compile(
    r"(?:บัญชีรายชื่อ|บช)"
)
RE_FORM_DISTRICT_EXPLICIT = re.compile(
    r"แบ่งเขต|แบบแบ่งเขต"
)
# Generic form number: matches "5-18", "5-17", "5/18" etc.
RE_FORM_NUMBER = re.compile(
    r"(?:ส\.?ส\.?|แบบ\s*ส\.?ส\.?)\s*5[-/]?1[78]"
)

# Unit number from filenames: "หน่วยที่ 1", "หน่วย3", "น.1", "น1"
RE_UNIT_NUM = re.compile(
    r"(?:หน่วย(?:ที่)?\s*(\d+)|น\.?\s*(\d+))"
)

# Special: "นอกเขต" (out-of-district / advance voting)
RE_NORK_KHET = re.compile(r"นอกเขต")


def parse_election_path(relpath: str) -> dict[str, Any]:
    """Parse metadata from a relative path under data/raw/.

    Args:
        relpath: Relative path from data/raw/ root (or absolute — province is first Thai-named segment).

    Returns:
        Dict with keys: province, constituency, amphoe, tambon, form_type,
        unit_number, is_advance_voting, raw_path.
    """
    result: dict[str, Any] = {
        "province": None,
        "constituency": None,
        "amphoe": None,
        "tambon": None,
        "form_type": None,
        "unit_number": None,
        "is_advance_voting": False,
        "raw_path": relpath,
    }

    # Normalize path separators
    clean = relpath.replace("\\", "/")
    parts = [p for p in clean.split("/") if p]

    if not parts:
        return result

    # First segment is typically the province name
    # Skip "data", "raw" prefixes if present
    start_idx = 0
    for i, p in enumerate(parts):
        if p.lower() in ("data", "raw"):
            start_idx = i + 1
            continue
        break
    if start_idx < len(parts):
        # Province: first segment that isn't data/raw and doesn't match a known pattern
        candidate = parts[start_idx]
        if not RE_CONSTITUENCY.search(candidate):
            result["province"] = candidate
            start_idx += 1

    # Parse each path segment (strip .pdf from filename segments)
    for raw_seg in parts[start_idx:]:
        seg = re.sub(r"\.pdf$", "", raw_seg, flags=re.IGNORECASE)

        # Constituency
        m = RE_CONSTITUENCY.search(seg)
        if m and result["constituency"] is None:
            result["constituency"] = int(m.group(1))
            continue

        # Advance voting
        if RE_NORK_KHET.search(seg):
            result["is_advance_voting"] = True

        # Amphoe
        m = RE_AMPHOE.search(seg)
        if m and result["amphoe"] is None:
            result["amphoe"] = m.group(1).strip()
            continue

        # Tambon
        m = RE_TAMBON.search(seg)
        if m and result["tambon"] is None:
            result["tambon"] = m.group(1).strip()
            continue

        # Form type (from directory name)
        if result["form_type"] is None:
            if RE_FORM_PARTYLIST.search(seg):
                result["form_type"] = "partylist"
            elif RE_FORM_DISTRICT_EXPLICIT.search(seg):
                result["form_type"] = "district"
            elif RE_FORM_NUMBER.search(seg) and not RE_FORM_PARTYLIST.search(seg):
                # Has form number but no partylist marker → district
                result["form_type"] = "district"

    # Unit number: search filename and last few segments
    filename = re.sub(r"\.pdf$", "", parts[-1] if parts else "", flags=re.IGNORECASE)
    search_parts = parts[-3:] if len(parts) >= 3 else parts
    for seg in search_parts:
        m = RE_UNIT_NUM.search(seg)
        if m:
            result["unit_number"] = int(m.group(1) or m.group(2))
            break

    # If form type still ambiguous, check filename
    if result["form_type"] is None:
        if RE_FORM_PARTYLIST.search(filename):
            result["form_type"] = "partylist"
        elif RE_FORM_DISTRICT_EXPLICIT.search(filename):
            result["form_type"] = "district"
        elif RE_FORM_NUMBER.search(filename):
            result["form_type"] = "district"

    # Fallback: try to extract tambon from bare filename like "พระบาท ชุดที่56.pdf"
    if result["tambon"] is None and filename:
        m = RE_TAMBON_FROM_FILENAME.search(filename)
        if m:
            result["tambon"] = m.group(1).strip()

    return result


def scan_directory(root: Path) -> list[dict[str, Any]]:
    """Scan a directory tree for PDFs and parse each path."""
    pdfs = sorted(root.rglob("*.pdf"))
    results = []
    for pdf in pdfs:
        try:
            relpath = str(pdf.relative_to(root))
        except ValueError:
            relpath = str(pdf)
        results.append(parse_election_path(relpath))
    return results


def to_csv(records: list[dict[str, Any]], output: io.TextIOBase | None = None) -> str:
    """Convert parsed records to CSV. Returns CSV string if output is None."""
    columns = [
        "province", "constituency", "amphoe", "tambon",
        "form_type", "unit_number", "is_advance_voting", "raw_path",
    ]
    buf = io.StringIO() if output is None else output
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for rec in records:
        writer.writerow(rec)
    if output is None:
        return buf.getvalue()
    return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Parse Thai election directory paths to extract structured metadata.",
    )
    ap.add_argument("path", help="PDF file path, directory path, or relative election path string.")
    ap.add_argument("--csv", default=None, help="Output CSV path. Use /dev/stdout for console.")
    ap.add_argument("--json", action="store_true", help="Output as JSON instead of human-readable.")
    args = ap.parse_args(argv)

    target = Path(args.path)

    if target.is_dir():
        records = scan_directory(target)
        if args.csv:
            csv_path = Path(args.csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(str(csv_path), "w", encoding="utf-8", newline="") as f:
                to_csv(records, f)
            print(f"Wrote {len(records)} records to {csv_path}")
        elif args.json:
            import json
            print(json.dumps(records, ensure_ascii=False, indent=2))
        else:
            # Summary
            provinces = {r["province"] for r in records if r["province"]}
            form_types = {r["form_type"] for r in records if r["form_type"]}
            parsed_const = sum(1 for r in records if r["constituency"] is not None)
            parsed_amphoe = sum(1 for r in records if r["amphoe"] is not None)
            parsed_tambon = sum(1 for r in records if r["tambon"] is not None)
            parsed_unit = sum(1 for r in records if r["unit_number"] is not None)
            print(f"Scanned {len(records)} PDFs")
            print(f"  Provinces: {sorted(provinces)}")
            print(f"  Form types: {sorted(form_types)}")
            print(f"  Parsed constituency: {parsed_const}/{len(records)}")
            print(f"  Parsed amphoe: {parsed_amphoe}/{len(records)}")
            print(f"  Parsed tambon: {parsed_tambon}/{len(records)}")
            print(f"  Parsed unit_number: {parsed_unit}/{len(records)}")
    elif target.is_file() or target.suffix == ".pdf":
        record = parse_election_path(args.path)
        if args.json:
            import json
            print(json.dumps(record, ensure_ascii=False, indent=2))
        else:
            for k, v in record.items():
                print(f"  {k}: {v}")
    else:
        # Treat as a raw path string
        record = parse_election_path(args.path)
        if args.json:
            import json
            print(json.dumps(record, ensure_ascii=False, indent=2))
        else:
            for k, v in record.items():
                print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
