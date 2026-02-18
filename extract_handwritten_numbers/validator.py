from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

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

        review_queue = []
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

        structured["validation"] = {
            "total_extractions": int(auto + review + manual),
            "auto_accepted": int(auto),
            "review_needed": int(review),
            "manual_entry": int(manual),
        }
        structured["review_queue"] = review_queue
        return structured

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


