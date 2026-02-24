"""Tests for scripts/cross_validate_pipelines.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from cross_validate_pipelines import (
    _extract_cv_fields,
    _extract_cv_table_total,
    _extract_gemini_stats,
    cross_validate,
)
from pipeline_io import parse_int


def test_parse_int_normal():
    assert parse_int("123") == 123
    assert parse_int(456) == 456
    assert parse_int("1,234") == 1234


def test_parse_int_none():
    assert parse_int(None) is None
    assert parse_int("") is None
    assert parse_int("abc") is None


def test_extract_cv_fields():
    result = {
        "documents": [{
            "fields": {
                "doc0_p1_field_1": {"value": "100"},
                "doc0_p1_field_5": {"value": "750"},
            },
        }],
    }
    fields = _extract_cv_fields(result)
    assert fields[1] == 100
    assert fields[5] == 750


def test_extract_cv_table_total():
    result = {
        "documents": [{
            "table": {
                "last_column_values": [
                    {"value": "100"},
                    {"value": "200"},
                    {"value": "300"},
                ],
            },
        }],
    }
    assert _extract_cv_table_total(result) == 300


def test_extract_gemini_stats():
    extraction = {
        "voter_stats": {
            "eligible_voters": 1000,
            "showed_up": 800,
            "valid_ballots": 750,
        },
    }
    stats = _extract_gemini_stats(extraction)
    assert stats["eligible_voters"] == 1000
    assert stats["valid_ballots"] == 750


def test_cross_validate_no_match():
    """No matching keys → no discrepancies."""
    result = cross_validate({"dir_a": {}}, {"other.pdf": {}})
    assert result == []


def test_cross_validate_field_mismatch():
    cv = {
        "test/form": {
            "documents": [{
                "fields": {
                    "doc0_p1_field_5": {"value": "750"},
                },
                "table": {"last_column_values": []},
            }],
        },
    }
    gemini = {
        "test/form.pdf": {
            "voter_stats": {
                "valid_ballots": 780,  # different from CV
            },
        },
    }
    discrepancies = cross_validate(cv, gemini)
    assert len(discrepancies) >= 1
    assert discrepancies[0]["check"] == "field_mismatch"
    assert discrepancies[0]["cv_value"] == 750
    assert discrepancies[0]["gemini_value"] == 780


def test_cross_validate_matching_values():
    cv = {
        "test/form": {
            "documents": [{
                "fields": {
                    "doc0_p1_field_5": {"value": "750"},
                },
                "table": {"last_column_values": []},
            }],
        },
    }
    gemini = {
        "test/form.pdf": {
            "voter_stats": {
                "valid_ballots": 750,
            },
        },
    }
    discrepancies = cross_validate(cv, gemini)
    assert discrepancies == []
