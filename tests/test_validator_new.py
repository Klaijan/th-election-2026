"""Tests for new validator functions: ballot fields, table sum, field completeness."""
import importlib.util
import sys
import types
from pathlib import Path

# Mock cv2 to avoid ImportError when __init__.py pulls in main.py
if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")
if "numpy" not in sys.modules:
    sys.modules["numpy"] = types.ModuleType("numpy")

from extract_handwritten_numbers.validator import Validator


# ---------------------------------------------------------------------------
# cross_validate_ballot_fields
# ---------------------------------------------------------------------------

def test_ballot_fields_all_correct():
    fields = {
        "doc0_p1_field_1": {"value": "1000"},   # eligible_voters
        "doc0_p1_field_2": {"value": "800"},     # voters_appeared
        "doc0_p1_field_3": {"value": "900"},     # ballots_allocated
        "doc0_p1_field_4": {"value": "800"},     # ballots_used
        "doc0_p1_field_5": {"value": "750"},     # valid_ballots
        "doc0_p1_field_6": {"value": "30"},      # invalid_ballots
        "doc0_p1_field_7": {"value": "20"},      # no_vote_ballots
        "doc0_p1_field_8": {"value": "100"},     # ballots_remaining
    }
    warnings = Validator.cross_validate_ballot_fields(fields)
    assert warnings == []


def test_ballot_fields_breakdown_mismatch():
    fields = {
        "doc0_p1_field_4": {"value": "800"},   # ballots_used
        "doc0_p1_field_5": {"value": "750"},   # valid
        "doc0_p1_field_6": {"value": "30"},    # invalid
        "doc0_p1_field_7": {"value": "10"},    # no_vote (wrong: 750+30+10=790≠800)
    }
    warnings = Validator.cross_validate_ballot_fields(fields)
    checks = [w["check"] for w in warnings]
    assert "ballot_breakdown_sum" in checks


def test_ballot_fields_allocation_mismatch():
    fields = {
        "doc0_p1_field_3": {"value": "1000"},  # ballots_allocated
        "doc0_p1_field_4": {"value": "800"},   # ballots_used
        "doc0_p1_field_8": {"value": "100"},   # remaining (800+100=900≠1000)
    }
    warnings = Validator.cross_validate_ballot_fields(fields)
    checks = [w["check"] for w in warnings]
    assert "ballot_allocation_sum" in checks


def test_ballot_fields_voters_vs_ballots():
    fields = {
        "doc0_p1_field_2": {"value": "800"},   # voters_appeared
        "doc0_p1_field_4": {"value": "750"},   # ballots_used (mismatch)
    }
    warnings = Validator.cross_validate_ballot_fields(fields)
    checks = [w["check"] for w in warnings]
    assert "voters_vs_ballots" in checks


def test_ballot_fields_missing_values_skipped():
    fields = {
        "doc0_p1_field_5": {"value": "750"},
    }
    warnings = Validator.cross_validate_ballot_fields(fields)
    assert warnings == []


def test_ballot_fields_empty_fields():
    warnings = Validator.cross_validate_ballot_fields({})
    assert warnings == []


# ---------------------------------------------------------------------------
# cross_validate_table_sum
# ---------------------------------------------------------------------------

def test_table_sum_correct():
    col_vals = [
        {"value": "100"},
        {"value": "200"},
        {"value": "50"},
        {"value": "350"},  # total row
    ]
    warnings = Validator.cross_validate_table_sum(col_vals)
    assert warnings == []


def test_table_sum_mismatch():
    col_vals = [
        {"value": "100"},
        {"value": "200"},
        {"value": "50"},
        {"value": "400"},  # wrong total
    ]
    warnings = Validator.cross_validate_table_sum(col_vals)
    checks = [w["check"] for w in warnings]
    assert "table_column_sum" in checks


def test_table_sum_cross_check_with_valid_ballots():
    col_vals = [
        {"value": "100"},
        {"value": "200"},
        {"value": "300"},  # total
    ]
    fields = {
        "doc0_p1_field_5": {"value": "999"},  # valid_ballots != 300
    }
    warnings = Validator.cross_validate_table_sum(col_vals, fields)
    checks = [w["check"] for w in warnings]
    assert "table_total_vs_valid_ballots" in checks


def test_table_sum_too_few_rows():
    warnings = Validator.cross_validate_table_sum([{"value": "100"}])
    assert warnings == []


def test_table_sum_empty():
    warnings = Validator.cross_validate_table_sum([])
    assert warnings == []


# ---------------------------------------------------------------------------
# check_field_completeness
# ---------------------------------------------------------------------------

def test_completeness_all_8_fields():
    fields = {f"doc0_p1_field_{i}": {"value": str(i * 100)} for i in range(1, 9)}
    warnings = Validator.check_field_completeness(fields)
    assert warnings == []


def test_completeness_missing_fields():
    fields = {f"doc0_p1_field_{i}": {"value": str(i * 100)} for i in range(1, 7)}
    warnings = Validator.check_field_completeness(fields)
    checks = [w["check"] for w in warnings]
    assert "incomplete_fields" in checks
    assert "missing 2" in warnings[0]["detail"]


def test_completeness_no_fields_no_warning():
    """Zero fields detected = probably a continuation page, not a detection failure."""
    warnings = Validator.check_field_completeness({})
    assert warnings == []


# ---------------------------------------------------------------------------
# validate_results integration (sanity_warnings in output)
# ---------------------------------------------------------------------------

def test_validate_results_includes_sanity_warnings():
    structured = {
        "fields": {
            "doc0_p1_field_4": {"value": "800", "confidence": 0.9},
            "doc0_p1_field_5": {"value": "750", "confidence": 0.9},
            "doc0_p1_field_6": {"value": "30", "confidence": 0.9},
            "doc0_p1_field_7": {"value": "10", "confidence": 0.9},  # 750+30+10=790≠800
        },
        "table": {"last_column_values": []},
    }
    v = Validator()
    result = v.validate_results(structured)
    assert "sanity_warnings" in result["validation"]
    assert len(result["validation"]["sanity_warnings"]) > 0


def test_validate_results_clean_data_no_warnings():
    structured = {
        "fields": {
            "doc0_p1_field_1": {"value": "1000", "confidence": 0.95},
            "doc0_p1_field_2": {"value": "800", "confidence": 0.95},
            "doc0_p1_field_3": {"value": "900", "confidence": 0.95},
            "doc0_p1_field_4": {"value": "800", "confidence": 0.95},
            "doc0_p1_field_5": {"value": "750", "confidence": 0.95},
            "doc0_p1_field_6": {"value": "30", "confidence": 0.95},
            "doc0_p1_field_7": {"value": "20", "confidence": 0.95},
            "doc0_p1_field_8": {"value": "100", "confidence": 0.95},
        },
        "table": {
            "last_column_values": [
                {"value": "350", "confidence": 0.9},
                {"value": "400", "confidence": 0.9},
                {"value": "750", "confidence": 0.9},  # total = 350+400 = 750 = valid_ballots
            ],
        },
    }
    v = Validator()
    result = v.validate_results(structured)
    sanity = result["validation"]["sanity_warnings"]
    assert len(sanity) == 0
