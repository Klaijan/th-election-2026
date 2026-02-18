from extract_handwritten_numbers.validator import Validator


def test_ballot_sum_passes_when_correct():
    stats = {
        "valid_ballots": 500,
        "invalid_ballots": 10,
        "no_vote_ballots": 5,
        "ballots_used": 515,
        "total_candidate_votes": 490,
    }
    warnings = Validator.cross_validate_voter_stats(stats)
    assert warnings == []


def test_ballot_sum_fails_when_mismatch():
    stats = {
        "valid_ballots": 500,
        "invalid_ballots": 10,
        "no_vote_ballots": 5,
        "ballots_used": 600,
        "total_candidate_votes": 490,
    }
    warnings = Validator.cross_validate_voter_stats(stats)
    checks = [w["check"] for w in warnings]
    assert "ballot_sum" in checks


def test_votes_exceed_valid_triggers_warning():
    stats = {
        "valid_ballots": 300,
        "invalid_ballots": 0,
        "no_vote_ballots": 0,
        "ballots_used": 300,
        "total_candidate_votes": 999,
    }
    warnings = Validator.cross_validate_voter_stats(stats)
    checks = [w["check"] for w in warnings]
    assert "votes_exceed_valid" in checks


def test_missing_fields_skips_check():
    stats = {"valid_ballots": 100}
    warnings = Validator.cross_validate_voter_stats(stats)
    assert isinstance(warnings, list)
    assert warnings == []


def test_string_values_coerced():
    stats = {
        "valid_ballots": "400",
        "invalid_ballots": "20",
        "no_vote_ballots": "5",
        "ballots_used": "425",
        "total_candidate_votes": "380",
    }
    warnings = Validator.cross_validate_voter_stats(stats)
    assert warnings == []
