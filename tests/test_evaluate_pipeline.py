"""Tests for scripts/evaluate_pipeline.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from evaluate_pipeline import (
    FieldResult,
    FormResult,
    compare_form,
    compute_metrics,
    format_summary,
    load_ground_truth,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_gt_ann(
    rel_path: str = "test.pdf",
    page_num: int = 1,
    voter_stats: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    return {
        "rel_path": rel_path,
        "page_num": page_num,
        "source": "human",
        "annotated_at": "2026-01-01T00:00:00+00:00",
        "metadata": metadata or {},
        "voter_stats": voter_stats or {},
        "candidates": None,
        "flags": [],
    }


ALL_CORRECT_STATS = {
    "eligible_voters": 142,
    "voters_appeared": 94,
    "ballots_allocated": 140,
    "ballots_used": 94,
    "valid_ballots": 88,
    "invalid_ballots": 4,
    "no_vote_ballots": 2,
    "ballots_remaining": 46,
}


# ---------------------------------------------------------------------------
# Test: load_ground_truth
# ---------------------------------------------------------------------------

class TestLoadGroundTruth:
    def test_load_basic(self, tmp_path):
        gt_path = tmp_path / "gt.jsonl"
        ann = _make_gt_ann(voter_stats={"eligible_voters": 100})
        gt_path.write_text(json.dumps(ann) + "\n")

        gt = load_ground_truth(gt_path)
        assert ("test.pdf", 1) in gt
        assert gt[("test.pdf", 1)]["voter_stats"]["eligible_voters"] == 100

    def test_load_multiple(self, tmp_path):
        gt_path = tmp_path / "gt.jsonl"
        lines = [
            json.dumps(_make_gt_ann("a.pdf", 1, {"eligible_voters": 100})),
            json.dumps(_make_gt_ann("b.pdf", 2, {"eligible_voters": 200})),
        ]
        gt_path.write_text("\n".join(lines) + "\n")

        gt = load_ground_truth(gt_path)
        assert len(gt) == 2


# ---------------------------------------------------------------------------
# Test: compare_form
# ---------------------------------------------------------------------------

class TestCompareForm:
    def test_all_match(self):
        ann = _make_gt_ann(voter_stats=ALL_CORRECT_STATS)
        pred = dict(ALL_CORRECT_STATS)
        result = compare_form(ann, pred)
        assert result.all_correct is True
        assert len(result.fields) == 8
        assert all(f.correct for f in result.fields)

    def test_one_mismatch(self):
        ann = _make_gt_ann(voter_stats=ALL_CORRECT_STATS)
        pred = {**ALL_CORRECT_STATS, "eligible_voters": 999}
        result = compare_form(ann, pred)
        assert result.all_correct is False
        ev_field = next(f for f in result.fields if f.field == "eligible_voters")
        assert ev_field.correct is False
        assert ev_field.abs_error == abs(142 - 999)

    def test_gt_null_skipped(self):
        """Fields where GT is null should be excluded from comparison."""
        ann = _make_gt_ann(voter_stats={"eligible_voters": 100, "voters_appeared": None})
        pred = {"eligible_voters": 100, "voters_appeared": 50}
        result = compare_form(ann, pred)
        assert len(result.fields) == 1  # only eligible_voters
        assert result.all_correct is True

    def test_pred_null_is_wrong(self):
        """Missing prediction counts as incorrect."""
        ann = _make_gt_ann(voter_stats={"eligible_voters": 100})
        pred: dict = {}
        result = compare_form(ann, pred)
        assert len(result.fields) == 1
        assert result.fields[0].correct is False
        assert result.fields[0].abs_error is None

    def test_empty_gt(self):
        ann = _make_gt_ann(voter_stats={})
        result = compare_form(ann, {"eligible_voters": 100})
        assert result.all_correct is False  # no fields → not all_correct
        assert len(result.fields) == 0


# ---------------------------------------------------------------------------
# Test: compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_score(self):
        ann = _make_gt_ann(voter_stats=ALL_CORRECT_STATS)
        fr = compare_form(ann, ALL_CORRECT_STATS)
        metrics = compute_metrics([fr], total_gt_forms=1)
        assert metrics.form_exact_match == 1.0
        assert metrics.field_accuracy == 1.0
        assert metrics.mean_abs_error == 0.0
        assert metrics.max_abs_error == 0

    def test_mixed_results(self):
        # Form 1: perfect
        fr1 = compare_form(
            _make_gt_ann(voter_stats={"eligible_voters": 100}),
            {"eligible_voters": 100},
        )
        # Form 2: wrong
        fr2 = compare_form(
            _make_gt_ann("b.pdf", voter_stats={"eligible_voters": 100}),
            {"eligible_voters": 110},
        )
        metrics = compute_metrics([fr1, fr2], total_gt_forms=2)
        assert metrics.matched_forms == 2
        assert metrics.form_exact_match == 0.5
        assert metrics.field_accuracy == 0.5
        assert metrics.mean_abs_error == 10.0
        assert metrics.max_abs_error == 10

    def test_per_field_breakdown(self):
        ann = _make_gt_ann(voter_stats={"eligible_voters": 100, "valid_ballots": 80})
        pred = {"eligible_voters": 100, "valid_ballots": 70}
        fr = compare_form(ann, pred)
        metrics = compute_metrics([fr], total_gt_forms=1)
        assert "eligible_voters" in metrics.per_field
        assert metrics.per_field["eligible_voters"]["accuracy"] == 1.0
        assert metrics.per_field["valid_ballots"]["accuracy"] == 0.0
        assert metrics.per_field["valid_ballots"]["mean_abs_error"] == 10.0

    def test_stratified_by_ink_type(self):
        fr1 = FormResult(
            rel_path="a.pdf", page_num=1,
            fields=[FieldResult("eligible_voters", 100, 100, True, 0)],
            all_correct=True,
            metadata={"ink_type": "blue"},
        )
        fr2 = FormResult(
            rel_path="b.pdf", page_num=1,
            fields=[FieldResult("eligible_voters", 100, 50, False, 50)],
            all_correct=False,
            metadata={"ink_type": "black"},
        )
        metrics = compute_metrics([fr1, fr2], total_gt_forms=2)
        assert metrics.by_ink_type is not None
        assert metrics.by_ink_type["blue"]["form_exact_match"] == 1.0
        assert metrics.by_ink_type["black"]["form_exact_match"] == 0.0

    def test_no_stratification_single_group(self):
        """If all forms share same ink_type, by_ink_type should be None."""
        fr = FormResult(
            rel_path="a.pdf", page_num=1,
            fields=[FieldResult("eligible_voters", 100, 100, True, 0)],
            all_correct=True,
            metadata={"ink_type": "blue"},
        )
        metrics = compute_metrics([fr], total_gt_forms=1)
        assert metrics.by_ink_type is None

    def test_empty_input(self):
        metrics = compute_metrics([], total_gt_forms=0)
        assert metrics.total_forms == 0
        assert metrics.form_exact_match == 0.0
        assert metrics.field_accuracy == 0.0


# ---------------------------------------------------------------------------
# Test: format_summary
# ---------------------------------------------------------------------------

class TestFormatSummary:
    def test_contains_key_sections(self):
        fr = compare_form(
            _make_gt_ann(voter_stats=ALL_CORRECT_STATS),
            ALL_CORRECT_STATS,
        )
        metrics = compute_metrics([fr], total_gt_forms=1)
        summary = format_summary(metrics)
        assert "Pipeline Evaluation" in summary
        assert "100.0%" in summary
        assert "eligible_voters" in summary

    def test_empty_metrics(self):
        metrics = compute_metrics([], total_gt_forms=0)
        summary = format_summary(metrics)
        assert "0" in summary
