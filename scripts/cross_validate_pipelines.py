#!/usr/bin/env python3
"""Cross-validate CV pipeline results against Gemini Vision LLM extraction.

Compares field-level values from the two independent pipelines and flags
discrepancies. This catches errors that neither pipeline can detect alone.

Usage:
    python scripts/cross_validate_pipelines.py \
        --cv-root output/ \
        --gemini-manifest data/gemini_manifest.jsonl \
        --csv data/cross_validation_report.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

from pipeline_io import load_cv_results, load_gemini_manifest, parse_int

# Mapping from CV pipeline field index (1-based region ID suffix) to
# Gemini extraction voter_stats keys.
CV_FIELD_TO_GEMINI = {
    1: "eligible_voters",
    2: "showed_up",
    3: "ballots_received",
    4: "ballots_used",
    5: "valid_ballots",
    6: "invalid_ballots",
    7: "no_vote_ballots",
    8: None,  # ballots_remaining — not in Gemini schema
}

def _extract_cv_fields(result: dict[str, Any]) -> dict[int, int | None]:
    """Extract field values from CV result by field index (1-based)."""
    values: dict[int, int | None] = {}
    # Try documents[0].fields first (multi-doc), then top-level fields
    docs = result.get("documents", [])
    fields = {}
    if docs:
        fields = docs[0].get("fields", {})
    if not fields:
        fields = result.get("fields", {})

    for fid, item in fields.items():
        # Extract field index from region ID like "doc0_p1_field_5"
        for i in range(1, 9):
            if fid.endswith(f"field_{i}"):
                values[i] = parse_int(item.get("value"))
                break
    return values


def _extract_cv_table_total(result: dict[str, Any]) -> int | None:
    """Extract table total (last row of last column) from CV result."""
    docs = result.get("documents", [])
    table = {}
    if docs:
        table = docs[0].get("table", {})
    if not table:
        table = result.get("table", {})
    col_vals = table.get("last_column_values", []) or table.get("column_3_values", [])
    if not col_vals:
        return None
    last_row = col_vals[-1]
    return parse_int(last_row.get("value"))


def _extract_gemini_stats(extraction: dict[str, Any]) -> dict[str, int | None]:
    """Extract voter_stats and total_candidate_votes from Gemini extraction."""
    stats = extraction.get("voter_stats", {}) or {}
    result: dict[str, int | None] = {}
    for key in ["eligible_voters", "showed_up", "ballots_received", "ballots_used",
                 "valid_ballots", "invalid_ballots", "no_vote_ballots", "total_candidate_votes"]:
        result[key] = parse_int(stats.get(key))
    return result


def _extract_gemini_vote_total(extraction: dict[str, Any]) -> int | None:
    """Sum candidate votes from Gemini extraction."""
    candidates = extraction.get("candidates", []) or []
    if not candidates:
        return None
    total = 0
    for c in candidates:
        v = parse_int(c.get("votes"))
        if v is None:
            return None  # incomplete data
        total += v
    return total


def cross_validate(
    cv_results: dict[str, dict[str, Any]],
    gemini_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare CV and Gemini results. Returns list of discrepancy records."""
    discrepancies: list[dict[str, Any]] = []

    # Build lookup: try to match CV dir names to Gemini rel_paths
    # CV keys: "subdir/name" (from result.json location)
    # Gemini keys: "rel/path/to/file.pdf" (from manifest)
    # Heuristic: Gemini rel_path without .pdf extension should match CV dir
    gemini_by_stem: dict[str, tuple[str, dict[str, Any]]] = {}
    for rel, rec in gemini_results.items():
        stem = str(Path(rel).with_suffix(""))
        gemini_by_stem[stem] = (rel, rec)

    matched = 0
    for cv_dir, cv_result in cv_results.items():
        # Try exact stem match
        gemini_match = gemini_by_stem.get(cv_dir)
        if not gemini_match:
            continue

        matched += 1
        gemini_rel, gemini_rec = gemini_match
        gemini_ext = gemini_rec.get("extraction", gemini_rec)
        cv_fields = _extract_cv_fields(cv_result)
        cv_table_total = _extract_cv_table_total(cv_result)
        gemini_stats = _extract_gemini_stats(gemini_ext)
        gemini_vote_total = _extract_gemini_vote_total(gemini_ext)

        # Compare field-by-field
        for field_idx, gemini_key in CV_FIELD_TO_GEMINI.items():
            if gemini_key is None:
                continue
            cv_val = cv_fields.get(field_idx)
            gem_val = gemini_stats.get(gemini_key)
            if cv_val is None or gem_val is None:
                continue
            if cv_val != gem_val:
                discrepancies.append({
                    "source": cv_dir,
                    "check": "field_mismatch",
                    "field": gemini_key,
                    "field_index": field_idx,
                    "cv_value": cv_val,
                    "gemini_value": gem_val,
                    "diff": abs(cv_val - gem_val),
                })

        # Compare table total vs Gemini total_candidate_votes
        gem_total = gemini_stats.get("total_candidate_votes")
        if cv_table_total is not None and gem_total is not None:
            if cv_table_total != gem_total:
                discrepancies.append({
                    "source": cv_dir,
                    "check": "table_total_mismatch",
                    "field": "total_candidate_votes",
                    "field_index": None,
                    "cv_value": cv_table_total,
                    "gemini_value": gem_total,
                    "diff": abs(cv_table_total - gem_total),
                })

        # Compare Gemini internal: sum of candidate votes vs total_candidate_votes
        if gemini_vote_total is not None and gem_total is not None:
            if gemini_vote_total != gem_total:
                discrepancies.append({
                    "source": cv_dir,
                    "check": "gemini_candidate_sum_mismatch",
                    "field": "sum(candidates.votes) vs total_candidate_votes",
                    "field_index": None,
                    "cv_value": None,
                    "gemini_value": gem_total,
                    "diff": abs(gemini_vote_total - gem_total),
                })

    return discrepancies


def to_csv(records: list[dict[str, Any]], output: io.TextIOBase | None = None) -> str:
    columns = ["source", "check", "field", "field_index", "cv_value", "gemini_value", "diff"]
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
        description="Cross-validate CV pipeline vs Gemini Vision LLM extraction.",
    )
    ap.add_argument("--cv-root", required=True, help="CV pipeline output root (contains result.json files).")
    ap.add_argument("--gemini-manifest", required=True, help="Gemini extraction manifest JSONL path.")
    ap.add_argument("--csv", default=None, help="Output CSV path for discrepancy report.")
    ap.add_argument("--json", action="store_true", help="Output as JSON instead of summary.")
    args = ap.parse_args(argv)

    cv_root = Path(args.cv_root)
    gemini_manifest = Path(args.gemini_manifest)

    if not cv_root.exists():
        print(f"Error: CV root not found: {cv_root}", file=sys.stderr)
        return 1
    if not gemini_manifest.exists():
        print(f"Error: Gemini manifest not found: {gemini_manifest}", file=sys.stderr)
        return 1

    cv_results = load_cv_results(cv_root)
    gemini_results = load_gemini_manifest(gemini_manifest)

    print(f"Loaded {len(cv_results)} CV results, {len(gemini_results)} Gemini results")

    discrepancies = cross_validate(cv_results, gemini_results)

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(csv_path), "w", encoding="utf-8", newline="") as f:
            to_csv(discrepancies, f)
        print(f"Wrote {len(discrepancies)} discrepancies to {csv_path}")
    elif args.json:
        print(json.dumps(discrepancies, ensure_ascii=False, indent=2))
    else:
        if not discrepancies:
            print("No discrepancies found between CV and Gemini pipelines.")
        else:
            print(f"Found {len(discrepancies)} discrepancies:")
            by_check = {}
            for d in discrepancies:
                by_check.setdefault(d["check"], []).append(d)
            for check, items in sorted(by_check.items()):
                print(f"  {check}: {len(items)}")
                for item in items[:5]:
                    print(f"    {item['source']}: {item['field']} CV={item['cv_value']} Gemini={item['gemini_value']}")
                if len(items) > 5:
                    print(f"    ... and {len(items) - 5} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
