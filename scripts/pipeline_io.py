"""Shared I/O utilities for loading CV and Gemini pipeline outputs.

Avoids duplication across cross_validate_pipelines.py and
evaluate_pipeline.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Canonical GT field names (from config.BALLOT_FIELD_MAP_DISTRICT order)
# ---------------------------------------------------------------------------

GT_FIELDS = [
    "eligible_voters",
    "voters_appeared",
    "ballots_allocated",
    "ballots_used",
    "valid_ballots",
    "invalid_ballots",
    "no_vote_ballots",
    "ballots_remaining",
]

# Gemini voter_stats key → GT canonical name (only 2 differ).
GEMINI_TO_GT: dict[str, str] = {
    "showed_up": "voters_appeared",
    "ballots_received": "ballots_allocated",
}

# CV field index (1-based region ID suffix) → 0-based GT_FIELDS index.
_CV_FIELD_TO_GT_INDEX: dict[int, int] = {i: i - 1 for i in range(1, 9)}


def parse_int(raw: str | int | float | None) -> int | None:
    """Coerce to int, stripping non-digit chars. Returns None on failure."""
    if raw is None:
        return None
    s = "".join(c for c in str(raw) if c.isdigit())
    if not s:
        return None
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def load_gemini_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Load Gemini manifest JSONL. Returns {rel_path: full_record}.

    Each record contains the full JSONL line (rel_path, page_num,
    extraction, status, etc.). Only records with status="ok" are included.
    """
    results: dict[str, dict[str, Any]] = {}
    if not manifest_path.exists():
        return results
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("status") != "ok":
                    continue
                rel = rec.get("rel_path", "")
                if rel:
                    results[rel] = rec
            except (json.JSONDecodeError, KeyError):
                continue
    return results


def load_cv_results(cv_root: Path) -> dict[str, dict[str, Any]]:
    """Load CV pipeline result.json files. Returns {relative_dir: result_dict}."""
    results: dict[str, dict[str, Any]] = {}
    for result_json in sorted(cv_root.rglob("result.json")):
        try:
            data = json.loads(result_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        rel_dir = str(result_json.parent.relative_to(cv_root))
        results[rel_dir] = data
    return results


# ---------------------------------------------------------------------------
# Normalizers: convert pipeline output to canonical GT field names
# ---------------------------------------------------------------------------

def normalize_cv_to_gt(result: dict[str, Any]) -> dict[str, int | None]:
    """Extract CV fields → {gt_field_name: int | None}."""
    docs = result.get("documents", [])
    fields: dict[str, Any] = {}
    if docs:
        fields = docs[0].get("fields", {})
    if not fields:
        fields = result.get("fields", {})

    gt: dict[str, int | None] = {}
    for fid, item in fields.items():
        for i in range(1, 9):
            if fid.endswith(f"field_{i}"):
                config_idx = _CV_FIELD_TO_GT_INDEX[i]
                gt_name = GT_FIELDS[config_idx]
                gt[gt_name] = parse_int(item.get("value"))
                break
    return gt


def normalize_gemini_to_gt(extraction: dict[str, Any]) -> dict[str, int | None]:
    """Extract Gemini voter_stats → {gt_field_name: int | None}."""
    stats = extraction.get("voter_stats", {}) or {}
    gt: dict[str, int | None] = {}
    for gemini_key, raw in stats.items():
        if gemini_key == "total_candidate_votes":
            continue
        gt_name = GEMINI_TO_GT.get(gemini_key, gemini_key)
        if gt_name in GT_FIELDS:
            gt[gt_name] = parse_int(raw)
    return gt
