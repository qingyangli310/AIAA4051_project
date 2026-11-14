"""Verification format tests. These do not load or evaluate a model."""

import pytest

from src.models.output_parser import parse_verification


@pytest.mark.parametrize("raw_output", ["valid", "  VALID\n"])
def test_valid_prediction_has_no_error_fields(raw_output):
    assert parse_verification(raw_output) == {
        "raw_output": raw_output,
        "parse_success": True,
        "valid": True,
        "first_error_step": None,
        "reason": None,
    }


@pytest.mark.parametrize(
    "reason",
    [
        "obstacle_collision",
        "out_of_bounds",
        "invalid_action_token",
        "goal_not_reached",
    ],
)
def test_invalid_prediction_preserves_step_and_supported_reason(reason):
    raw_output = f"invalid | step=2 | reason={reason}"
    assert parse_verification(raw_output) == {
        "raw_output": raw_output,
        "parse_success": True,
        "valid": False,
        "first_error_step": 2,
        "reason": reason,
    }


def test_verification_normalizes_case_and_spacing_without_changing_raw_text():
    raw_output = " INVALID | STEP = 1 | REASON = OUT_OF_BOUNDS \n"
    assert parse_verification(raw_output) == {
        "raw_output": raw_output,
        "parse_success": True,
        "valid": False,
        "first_error_step": 1,
        "reason": "out_of_bounds",
    }


@pytest.mark.parametrize(
    "raw_output",
    [
        pytest.param("", id="empty-output"),
        pytest.param(" \n ", id="whitespace-only"),
        pytest.param("invalid", id="missing-diagnosis"),
        pytest.param("invalid | reason=out_of_bounds", id="missing-step"),
        pytest.param("invalid | step=2", id="missing-reason"),
        pytest.param("invalid | step=-1 | reason=out_of_bounds", id="negative-step"),
        pytest.param("invalid | step=1.5 | reason=out_of_bounds", id="fractional-step"),
        pytest.param("invalid | step=two | reason=out_of_bounds", id="word-step"),
        pytest.param("invalid | step=2 | reason=collision", id="unknown-reason"),
        pytest.param("valid | step=2 | reason=out_of_bounds", id="contradictory-valid"),
        pytest.param("The plan is valid", id="explanation-instead-of-label"),
        pytest.param("invalid | step=2 | reason=out_of_bounds because it exits", id="extra-prose"),
        pytest.param("invalid | step=2 | step=3 | reason=out_of_bounds", id="duplicate-field"),
        pytest.param("invalid | reason=out_of_bounds | step=2", id="wrong-field-order"),
    ],
)
def test_malformed_prediction_is_not_a_negative_judgment(raw_output):
    assert parse_verification(raw_output) == {
        "raw_output": raw_output,
        "parse_success": False,
        "valid": None,
        "first_error_step": None,
        "reason": None,
    }


@pytest.mark.parametrize("reason", ["out_of_bounds", "goal_not_reached"])
def test_zero_step_is_valid_under_zero_based_protocol(reason):
    result = parse_verification(f"invalid | step=0 | reason={reason}")
    assert result["parse_success"] is True
    assert result["valid"] is False
    assert result["first_error_step"] == 0
    assert result["reason"] == reason
