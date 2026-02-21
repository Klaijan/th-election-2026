#!/usr/bin/env python3
"""Evaluate pipeline accuracy against ground truth annotations.

Computes exact-match metrics, MAE, per-field breakdown, and stratified
metrics by ink type and province.

Usage:
    python scripts/evaluate_pipeline.py \
        --gt data/ground_truth/annotations.jsonl \
        --pipeline cv \
        --cv-root output/ \
        [--json metrics.json] [--csv detail.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline_io import (
    GT_FIELDS,
    load_cv_results,
    load_gemini_manifest,
    normalize_cv_to_gt,
    normalize_gemini_to_gt,
)


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass
class FieldResult:
    field: str
    gt_value: int | None
    pred_value: int | None
    correct: bool
    abs_error: int | None


@dataclass
class FormResult:
    rel_path: str
    page_num: int
    fields: list[FieldResult]
    all_correct: bool
    metadata: dict[str, Any]


@dataclass
class EvalMetrics:
    total_forms: int
    matched_forms: int
    form_exact_match: float
    total_fields: int
    correct_fields: int
    field_accuracy: float
    mean_abs_error: float
    max_abs_error: int
    per_field: dict[str, dict[str, Any]]
    by_ink_type: dict[str, dict[str, Any]] | None
    by_province: dict[str, dict[str, Any]] | None


# ---------------------------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------------------------

def load_ground_truth(gt_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Load annotations JSONL. Returns {(rel_path, page_num): annotation}."""
    results: dict[tuple[str, int], dict[str, Any]] = {}
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["rel_path"], rec["page_num"])
            results[key] = rec
    return results


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_form(
    gt_ann: dict[str, Any],
    pred_stats: dict[str, int | None],
) -> FormResult:
    """Compare a single form's predicted values against ground truth."""
    gt_stats = gt_ann.get("voter_stats", {})
    fields: list[FieldResult] = []

    for fname in GT_FIELDS:
        gt_val = gt_stats.get(fname)
        pred_val = pred_stats.get(fname)

        # Skip fields where GT is null (unannotated)
        if gt_val is None:
            continue

        if pred_val is None:
            fields.append(FieldResult(
                field=fname, gt_value=gt_val, pred_value=None,
                correct=False, abs_error=None,
            ))
        else:
            correct = gt_val == pred_val
            abs_err = abs(gt_val - pred_val)
            fields.append(FieldResult(
                field=fname, gt_value=gt_val, pred_value=pred_val,
                correct=correct, abs_error=abs_err,
            ))

    all_correct = len(fields) > 0 and all(fr.correct for fr in fields)

    return FormResult(
        rel_path=gt_ann["rel_path"],
        page_num=gt_ann["page_num"],
        fields=fields,
        all_correct=all_correct,
        metadata=gt_ann.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _compute_group_metrics(form_results: list[FormResult]) -> dict[str, Any]:
    """Compute aggregate metrics for a group of form results."""
    if not form_results:
        return {
            "forms": 0, "form_exact_match": 0.0,
            "fields": 0, "correct": 0, "field_accuracy": 0.0,
            "mean_abs_error": 0.0, "max_abs_error": 0,
        }

    total_forms = len(form_results)
    exact_match = sum(1 for fr in form_results if fr.all_correct)

    all_fields: list[FieldResult] = []
    for fr in form_results:
        all_fields.extend(fr.fields)

    total_fields = len(all_fields)
    correct_fields = sum(1 for f in all_fields if f.correct)
    errors = [f.abs_error for f in all_fields if f.abs_error is not None and not f.correct]
    mae = sum(errors) / len(errors) if errors else 0.0
    max_err = max(errors) if errors else 0

    return {
        "forms": total_forms,
        "form_exact_match": exact_match / total_forms if total_forms else 0.0,
        "fields": total_fields,
        "correct": correct_fields,
        "field_accuracy": correct_fields / total_fields if total_fields else 0.0,
        "mean_abs_error": round(mae, 2),
        "max_abs_error": max_err,
    }


def compute_metrics(
    form_results: list[FormResult],
    total_gt_forms: int,
) -> EvalMetrics:
    """Compute full evaluation metrics from form results."""
    group = _compute_group_metrics(form_results)

    # Per-field breakdown
    per_field: dict[str, dict[str, Any]] = {}
    by_field: dict[str, list[FieldResult]] = {}
    for fr in form_results:
        for fld in fr.fields:
            by_field.setdefault(fld.field, []).append(fld)

    for fname, field_results in by_field.items():
        total = len(field_results)
        correct = sum(1 for f in field_results if f.correct)
        errors = [f.abs_error for f in field_results if f.abs_error is not None and not f.correct]
        per_field[fname] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "mean_abs_error": round(sum(errors) / len(errors), 2) if errors else 0.0,
            "max_abs_error": max(errors) if errors else 0,
        }

    # Stratified by ink_type
    by_ink: dict[str, list[FormResult]] = {}
    for fr in form_results:
        ink = fr.metadata.get("ink_type") or "unknown"
        by_ink.setdefault(ink, []).append(fr)
    by_ink_metrics = {k: _compute_group_metrics(v) for k, v in by_ink.items()} if len(by_ink) > 1 else None

    # Stratified by province
    by_prov: dict[str, list[FormResult]] = {}
    for fr in form_results:
        prov = fr.metadata.get("province") or "unknown"
        by_prov.setdefault(prov, []).append(fr)
    by_prov_metrics = {k: _compute_group_metrics(v) for k, v in by_prov.items()} if len(by_prov) > 1 else None

    return EvalMetrics(
        total_forms=total_gt_forms,
        matched_forms=group["forms"],
        form_exact_match=group["form_exact_match"],
        total_fields=group["fields"],
        correct_fields=group["correct"],
        field_accuracy=group["field_accuracy"],
        mean_abs_error=group["mean_abs_error"],
        max_abs_error=group["max_abs_error"],
        per_field=per_field,
        by_ink_type=by_ink_metrics,
        by_province=by_prov_metrics,
    )


# ---------------------------------------------------------------------------
# Pipeline data loading
# ---------------------------------------------------------------------------

def load_pipeline_predictions(
    pipeline: str,
    cv_root: Path | None,
    gemini_manifest: Path | None,
) -> dict[tuple[str, int], dict[str, int | None]]:
    """Load predictions from chosen pipeline. Returns {(stem, page_num): gt_fields}."""
    preds: dict[tuple[str, int], dict[str, int | None]] = {}

    if pipeline == "cv" and cv_root:
        for cv_dir, result in load_cv_results(cv_root).items():
            preds[(cv_dir, 1)] = normalize_cv_to_gt(result)

    elif pipeline == "gemini" and gemini_manifest:
        for rel, rec in load_gemini_manifest(gemini_manifest).items():
            extraction = rec.get("extraction", {})
            page_num = rec.get("page_num", 1)
            stem = str(Path(rel).with_suffix(""))
            preds[(stem, page_num)] = normalize_gemini_to_gt(extraction)

    return preds


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def format_summary(metrics: EvalMetrics) -> str:
    """Format metrics as human-readable summary."""
    lines = [
        "=== Pipeline Evaluation ===",
        f"GT forms: {metrics.total_forms}  |  Matched: {metrics.matched_forms}",
        f"Form exact match: {metrics.form_exact_match:.1%}",
        f"Field accuracy:   {metrics.field_accuracy:.1%} ({metrics.correct_fields}/{metrics.total_fields})",
        f"MAE (errors only): {metrics.mean_abs_error:.2f}  |  Max error: {metrics.max_abs_error}",
        "",
        "--- Per-field breakdown ---",
    ]
    for fname in GT_FIELDS:
        if fname in metrics.per_field:
            pf = metrics.per_field[fname]
            lines.append(f"  {fname:24s} {pf['accuracy']:6.1%} ({pf['correct']}/{pf['total']})  MAE={pf['mean_abs_error']:.1f}")

    if metrics.by_ink_type:
        lines.append("")
        lines.append("--- By ink type ---")
        for ink, m in sorted(metrics.by_ink_type.items()):
            lines.append(f"  {ink:12s} form_exact={m['form_exact_match']:.1%}  field_acc={m['field_accuracy']:.1%}  n={m['forms']}")

    if metrics.by_province:
        lines.append("")
        lines.append("--- By province ---")
        for prov, m in sorted(metrics.by_province.items()):
            lines.append(f"  {prov:20s} form_exact={m['form_exact_match']:.1%}  field_acc={m['field_accuracy']:.1%}  n={m['forms']}")

    return "\n".join(lines)


def write_detail_csv(form_results: list[FormResult], path: Path) -> None:
    """Write per-field detail CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["rel_path", "page_num", "field", "gt_value", "pred_value", "correct", "abs_error"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for fr in form_results:
            for fld in fr.fields:
                writer.writerow({
                    "rel_path": fr.rel_path,
                    "page_num": fr.page_num,
                    "field": fld.field,
                    "gt_value": fld.gt_value,
                    "pred_value": fld.pred_value,
                    "correct": fld.correct,
                    "abs_error": fld.abs_error,
                })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate pipeline accuracy against ground truth.")
    ap.add_argument("--gt", required=True, help="Ground truth annotations JSONL.")
    ap.add_argument("--pipeline", required=True, choices=["cv", "gemini"], help="Pipeline to evaluate.")
    ap.add_argument("--cv-root", default=None, help="CV pipeline output root.")
    ap.add_argument("--gemini-manifest", default=None, help="Gemini manifest JSONL.")
    ap.add_argument("--json", default=None, dest="json_out", help="Output metrics as JSON.")
    ap.add_argument("--csv", default=None, dest="csv_out", help="Output per-field detail CSV.")
    args = ap.parse_args(argv)

    gt_path = Path(args.gt)
    if not gt_path.exists():
        print(f"Error: GT not found: {gt_path}", file=sys.stderr)
        return 1

    gt = load_ground_truth(gt_path)
    preds = load_pipeline_predictions(
        args.pipeline,
        Path(args.cv_root) if args.cv_root else None,
        Path(args.gemini_manifest) if args.gemini_manifest else None,
    )

    # Match GT to predictions
    form_results: list[FormResult] = []
    for (rel_path, page_num), ann in gt.items():
        stem = str(Path(rel_path).with_suffix(""))
        pred_key = (stem, page_num)
        pred_stats = preds.get(pred_key, {})
        fr = compare_form(ann, pred_stats)
        form_results.append(fr)

    metrics = compute_metrics(form_results, total_gt_forms=len(gt))

    # Output
    if args.json_out:
        import dataclasses
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(dataclasses.asdict(metrics), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote metrics to {out_path}")

    if args.csv_out:
        write_detail_csv(form_results, Path(args.csv_out))
        print(f"Wrote detail CSV to {args.csv_out}")

    print(format_summary(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
