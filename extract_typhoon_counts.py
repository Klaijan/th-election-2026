#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_THAI_DIGITS = str.maketrans(
    {
        "๐": "0",
        "๑": "1",
        "๒": "2",
        "๓": "3",
        "๔": "4",
        "๕": "5",
        "๖": "6",
        "๗": "7",
        "๘": "8",
        "๙": "9",
    }
)


def _to_ascii_digits(s: str) -> str:
    return (s or "").translate(_THAI_DIGITS)


def _parse_int(s: str) -> int | None:
    s = _to_ascii_digits(s)
    s = re.sub(r"[^0-9]", "", s)
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


_RE_REL_PATH = re.compile(r"^<!--\s*rel_path=(.*?)\s*-->$")
_RE_SCHEMA_PREFIX = re.compile(r"^\s*([0-9๐๑๒๓๔๕๖๗๘๙]+(?:\.[0-9๐๑๒๓๔๕๖๗๘๙]+)*)\s+")
_RE_COUNT = re.compile(r"จำนวน\s+([0-9๐๑๒๓๔๕๖๗๘๙]+)\s*(คน|บัตร)\b")


@dataclass(frozen=True)
class CountItem:
    order: int
    schema: str
    label: str
    value: int
    unit: str
    raw_line: str


def iter_md_files(md_root: Path) -> Iterable[Path]:
    for p in sorted(md_root.rglob("*.md")):
        if p.is_file():
            yield p


def parse_counts_from_md_text(text: str) -> tuple[str, list[CountItem]]:
    rel_path = ""
    items: list[CountItem] = []
    order = 0

    for raw in (text or "").splitlines():
        line = raw.strip("\n")
        if not line.strip():
            continue

        m_rel = _RE_REL_PATH.match(line.strip())
        if m_rel and not rel_path:
            rel_path = m_rel.group(1).strip()
            continue

        # Ignore other HTML comments (fingerprint/model/page markers).
        if line.strip().startswith("<!--") and line.strip().endswith("-->"):
            continue

        for m in _RE_COUNT.finditer(line):
            raw_num = m.group(1)
            unit = m.group(2)
            value = _parse_int(raw_num)
            if value is None:
                continue

            label_raw = line[: m.start()].strip()

            schema = ""
            m_schema = _RE_SCHEMA_PREFIX.match(label_raw)
            if m_schema:
                schema = _to_ascii_digits(m_schema.group(1))
                label_raw = label_raw[m_schema.end() :].strip()

            # Normalize common punctuation/whitespace.
            label = re.sub(r"\s+", " ", label_raw).strip(" :-–—\t")

            items.append(
                CountItem(
                    order=order,
                    schema=schema,
                    label=label,
                    value=int(value),
                    unit=unit,
                    raw_line=line.strip(),
                )
            )
            order += 1

    return rel_path, items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract 'จำนวน <n> คน/บัตร' counts from Typhoon OCR Markdown outputs.")
    ap.add_argument("--md-root", required=True, help="Root folder containing Typhoon .md outputs (recursive).")
    ap.add_argument("--out-jsonl", required=True, help="Output JSONL path (one row per input .md).")
    ap.add_argument(
        "--kind",
        choices=["", "district", "partylist"],
        default="",
        help="Optional filter by rel_path containing district/partylist. Default: no filter.",
    )
    args = ap.parse_args(argv)

    md_root = Path(args.md_root)
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        for md_path in iter_md_files(md_root):
            text = md_path.read_text(encoding="utf-8", errors="replace")
            rel_path, items = parse_counts_from_md_text(text)
            rel = rel_path or str(md_path.relative_to(md_root))

            if args.kind:
                parts = {p.lower() for p in Path(rel).parts}
                if str(args.kind).lower() not in parts:
                    continue

            f.write(
                json.dumps(
                    {
                        "rel_path": rel,
                        "md_path": str(md_path),
                        "counts": [
                            {
                                "order": it.order,
                                "schema": it.schema,
                                "label": it.label,
                                "value": it.value,
                                "unit": it.unit,
                                "raw_line": it.raw_line,
                            }
                            for it in items
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            rows += 1

    print(f"Done. Wrote {rows} row(s) to: {out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


