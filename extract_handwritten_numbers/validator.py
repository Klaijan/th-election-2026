from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from . import config

log = logging.getLogger("extract_handwritten_numbers")


class Validator:
    """
    Apply lightweight business validation and confidence-based triage.
    """

    def __init__(
        self,
        *,
        field_range: Tuple[int, int] = config.FIELD_VALUE_RANGE,
        table_range: Tuple[int, int] = config.TABLE_VALUE_RANGE,
        auto_threshold: float = config.OCR_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.field_range = (int(field_range[0]), int(field_range[1]))
        self.table_range = (int(table_range[0]), int(table_range[1]))
        self.auto_threshold = float(auto_threshold)

    def validate_results(self, structured: Dict[str, Any]) -> Dict[str, Any]:
        fields = structured.get("fields", {}) or {}
        table = structured.get("table", {}) or {}

        review_queue: List[Dict[str, Any]] = []
        auto = 0
        review = 0
        manual = 0

        for fid, item in fields.items():
            value = str(item.get("value", "") or "")
            conf = float(item.get("confidence", 0.0) or 0.0)
            vr = self._validate_range(value, self.field_range)
            status = self._status(conf)
            item["status"] = status
            item["validation"] = {"range": vr}
            if status == "auto_accepted" and vr == "pass":
                auto += 1
            elif status == "manual_entry":
                manual += 1
                review_queue.append({"id": fid, "value": value, "confidence": conf, "reason": "low_confidence"})
            else:
                review += 1
                reason = "low_confidence" if status != "auto_accepted" else "validation_warning"
                review_queue.append({"id": fid, "value": value, "confidence": conf, "reason": reason})

        col_vals = list(table.get("last_column_values", []) or table.get("column_3_values", []) or [])
        for cell in col_vals:
            cid = str(cell.get("id") or cell.get("cell_id") or "")
            value = str(cell.get("value", "") or "")
            conf = float(cell.get("confidence", 0.0) or 0.0)
            vr = self._validate_range(value, self.table_range)
            status = self._status(conf)
            cell["status"] = status
            cell["validation"] = {"range": vr}
            if status == "auto_accepted" and vr == "pass":
                auto += 1
            elif status == "manual_entry":
                manual += 1
                review_queue.append({"id": cid, "value": value, "confidence": conf, "reason": "low_confidence"})
            else:
                review += 1
                reason = "low_confidence" if status != "auto_accepted" else "validation_warning"
                if vr != "pass":
                    reason = "outlier_value"
                review_queue.append({"id": cid, "value": value, "confidence": conf, "reason": reason})

        # Cross-validate ballot fields (positional from CV pipeline)
        ballot_warnings = self.cross_validate_ballot_fields(fields)
        # Cross-validate table column sum
        table_warnings = self.cross_validate_table_sum(col_vals, fields)
        # Check for incomplete field extraction
        completeness_warnings = self.check_field_completeness(fields)

        sanity_warnings = ballot_warnings + table_warnings + completeness_warnings
        for w in sanity_warnings:
            review_queue.append({
                "id": w.get("check", "sanity_check"),
                "value": "",
                "confidence": 0.0,
                "reason": f"sanity_check: {w['check']} — {w['detail']}",
            })

        structured["validation"] = {
            "total_extractions": int(auto + review + manual),
            "auto_accepted": int(auto),
            "review_needed": int(review),
            "manual_entry": int(manual),
            "sanity_warnings": sanity_warnings,
        }
        structured["review_queue"] = review_queue
        return structured

    @staticmethod
    def cross_validate_ballot_fields(fields: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check arithmetic consistency of positional ballot fields from CV pipeline.

        Uses BALLOT_FIELD_MAP_DISTRICT to map field_N region IDs to semantic names,
        then verifies:
          1. valid + invalid + no_vote == ballots_used
          2. ballots_used + ballots_remaining == ballots_allocated
          3. voters_appeared == ballots_used
        """
        warnings: List[Dict[str, Any]] = []
        field_map = config.BALLOT_FIELD_MAP_DISTRICT

        # Build semantic lookup: try "field_N" keys (1-based region IDs from extractor)
        semantic: Dict[str, int | None] = {}
        for idx, name in field_map.items():
            val = None
            # Region IDs may have prefixes like "doc0_p1_field_1" — try suffix match
            for fid, item in fields.items():
                # Match "field_{idx+1}" at the end of the region ID
                if fid.endswith(f"field_{idx + 1}"):
                    raw = str(item.get("value", "") or "")
                    digits = "".join(c for c in raw if c.isdigit())
                    if digits:
                        try:
                            val = int(digits)
                        except (ValueError, TypeError):
                            pass
                    break
            semantic[name] = val

        def _g(k: str) -> int | None:
            return semantic.get(k)

        valid = _g("valid_ballots")
        invalid = _g("invalid_ballots")
        no_vote = _g("no_vote_ballots")
        used = _g("ballots_used")
        remaining = _g("ballots_remaining")
        allocated = _g("ballots_allocated")
        appeared = _g("voters_appeared")

        # Check 1: valid + invalid + no_vote == ballots_used
        if valid is not None and invalid is not None and no_vote is not None and used is not None:
            expected = valid + invalid + no_vote
            if expected != used:
                warnings.append({
                    "check": "ballot_breakdown_sum",
                    "detail": (
                        f"valid({valid}) + invalid({invalid}) + no_vote({no_vote}) "
                        f"= {expected} != ballots_used({used})"
                    ),
                })

        # Check 2: ballots_used + ballots_remaining == ballots_allocated
        if used is not None and remaining is not None and allocated is not None:
            expected = used + remaining
            if expected != allocated:
                warnings.append({
                    "check": "ballot_allocation_sum",
                    "detail": (
                        f"ballots_used({used}) + remaining({remaining}) "
                        f"= {expected} != ballots_allocated({allocated})"
                    ),
                })

        # Check 3: voters_appeared == ballots_used
        if appeared is not None and used is not None:
            if appeared != used:
                warnings.append({
                    "check": "voters_vs_ballots",
                    "detail": f"voters_appeared({appeared}) != ballots_used({used})",
                })

        return warnings

    @staticmethod
    def cross_validate_table_sum(
        col_vals: List[Dict[str, Any]],
        fields: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Check that candidate vote rows sum to the total row, and cross-check with valid_ballots.

        col_vals: list of table cell dicts with "value" keys (last column).
        fields: optional field dict for cross-checking table total against valid_ballots.
        """
        warnings: List[Dict[str, Any]] = []
        if not col_vals or len(col_vals) < 2:
            return warnings

        def _parse(raw: str) -> int | None:
            digits = "".join(c for c in (raw or "") if c.isdigit())
            if not digits:
                return None
            try:
                return int(digits)
            except (ValueError, TypeError):
                return None

        # Last row is expected to be the total row ("รวมคะแนนทั้งสิ้น")
        candidate_rows = col_vals[:-1]
        total_row = col_vals[-1]

        total_val = _parse(str(total_row.get("value", "") or ""))
        row_values = [_parse(str(r.get("value", "") or "")) for r in candidate_rows]

        # Sum check: all candidate rows should sum to total
        if total_val is not None and all(v is not None for v in row_values):
            row_sum = sum(v for v in row_values if v is not None)
            if row_sum != total_val:
                warnings.append({
                    "check": "table_column_sum",
                    "detail": (
                        f"sum of {len(candidate_rows)} candidate rows = {row_sum} "
                        f"!= total_row({total_val})"
                    ),
                })

        # Cross-check: table total == valid_ballots from fields
        if total_val is not None and fields:
            field_map = config.BALLOT_FIELD_MAP_DISTRICT
            valid_idx = None
            for idx, name in field_map.items():
                if name == "valid_ballots":
                    valid_idx = idx
                    break
            if valid_idx is not None:
                for fid, item in (fields or {}).items():
                    if fid.endswith(f"field_{valid_idx + 1}"):
                        raw = str(item.get("value", "") or "")
                        valid_val = _parse(raw)
                        if valid_val is not None and valid_val != total_val:
                            warnings.append({
                                "check": "table_total_vs_valid_ballots",
                                "detail": (
                                    f"table_total({total_val}) != "
                                    f"valid_ballots(field_{valid_idx + 1}={valid_val})"
                                ),
                            })
                        break

        return warnings

    @staticmethod
    def check_field_completeness(fields: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check whether the expected number of ballot fields were detected.

        The district form (สส.5/18) should have 8 dotted-line fields.
        If fewer are extracted, flag as incomplete.
        """
        warnings: List[Dict[str, Any]] = []
        expected = len(config.BALLOT_FIELD_MAP_DISTRICT)
        if expected == 0:
            return warnings

        # Count field_N entries (match region IDs ending with field_<digit>)
        detected = sum(1 for fid in fields if "field_" in fid)

        if detected == 0:
            # No fields at all — may be a continuation page or detection failure
            return warnings

        if detected < expected:
            warnings.append({
                "check": "incomplete_fields",
                "detail": (
                    f"detected {detected} field(s) but expected {expected} "
                    f"(missing {expected - detected})"
                ),
            })

        return warnings

    @staticmethod
    def cross_validate_voter_stats(stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check arithmetic consistency of voter statistics.

        Expected: valid_ballots + invalid_ballots + no_vote_ballots == ballots_used
        Expected: total_candidate_votes <= valid_ballots
        """
        warnings: List[Dict[str, Any]] = []

        def _int(key: str) -> int | None:
            v = stats.get(key)
            if v is None:
                return None
            try:
                return int(v)
            except (ValueError, TypeError):
                return None

        valid = _int("valid_ballots")
        invalid = _int("invalid_ballots")
        no_vote = _int("no_vote_ballots")
        used = _int("ballots_used")
        total_votes = _int("total_candidate_votes")

        if valid is not None and invalid is not None and no_vote is not None and used is not None:
            expected = valid + invalid + no_vote
            if expected != used:
                warnings.append({
                    "check": "ballot_sum",
                    "detail": f"valid({valid}) + invalid({invalid}) + no_vote({no_vote}) = {expected} != ballots_used({used})",
                })

        if total_votes is not None and valid is not None:
            if total_votes > valid:
                warnings.append({
                    "check": "votes_exceed_valid",
                    "detail": f"total_candidate_votes({total_votes}) > valid_ballots({valid})",
                })

        return warnings

    def _status(self, confidence: float) -> str:
        if float(confidence) >= float(self.auto_threshold):
            return "auto_accepted"
        if float(confidence) >= 0.40:
            return "review_queue"
        return "manual_entry"

    @staticmethod
    def _parse_int(value: str) -> int | None:
        s = "".join(c for c in (value or "") if c.isdigit())
        if not s:
            return None
        try:
            return int(s)
        except Exception:
            return None

    def _validate_range(self, value: str, r: Tuple[int, int]) -> str:
        v = self._parse_int(value)
        if v is None:
            return "warning: empty_or_non_numeric"
        lo, hi = int(r[0]), int(r[1])
        if v < lo or v > hi:
            return f"warning: out_of_range ({v} not in [{lo},{hi}])"
        return "pass"
