"""Run with pytest; these tests never download/load a model."""

from copy import deepcopy
import json

import pytest

from scripts.member_b_day13 import DEFAULT_SAMPLES, check_sample
from src.models.day13_contract import convert_sample
from src.models.output_parser import parse_plan, parse_verification


def samples():
    return [json.loads(line) for line in DEFAULT_SAMPLES.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_eight_samples_roundtrip_with_explicit_expected_targets():
    expected = [
        "valid",
        "invalid | step=4 | reason=obstacle_collision",
        "invalid | step=1 | reason=out_of_bounds",
        "invalid | step=1 | reason=invalid_action_token",
        "invalid | step=1 | reason=goal_not_reached",
        "invalid | step=0 | reason=out_of_bounds",
        "invalid | step=1 | reason=obstacle_collision",
        "invalid | step=0 | reason=obstacle_collision",
    ]
    rows = samples()
    assert len(rows) == 8
    assert len({row["sample_id"] for row in rows}) == 8
    for row, target in zip(rows, expected):
        generation, verification, case = check_sample(row)
        assert case["passed"], case
        assert verification["target_text"] == target
        assert generation["expected_plan"] == row["correct_plan"]


def test_goal_failure_maps_success_not_movement_validity():
    row = samples()[4]
    assert row["executor_label"]["valid"] is True
    _, record = convert_sample(row)
    assert record["expected_verification"]["valid"] is False


def test_unknown_candidate_token_is_retained_and_correct_answer_is_not_in_prompt():
    _, record = convert_sample(samples()[3])
    assert record["candidate_plan"] == ["up", "teleport"]
    assert "Candidate path: up, teleport\n" in record["input_text"]
    assert "up, up, right, right" not in record["input_text"]
    assert record["target_text"] not in record["input_text"]


@pytest.mark.parametrize("text,length,accepted", [
    ("invalid | step=999 | reason=out_of_bounds", 3, False),
    ("invalid | step=3 | reason=obstacle_collision", 3, False),
    ("invalid | step=2 | reason=goal_not_reached", 3, False),
    ("invalid | step=3 | reason=goal_not_reached", 3, True),
    ("invalid | step=0 | reason=goal_not_reached", 0, True),
    ("invalid | step=0 | reason=invalid_action_token", 0, False),
    (" INVALID\n| STEP = 0\n| REASON = OUT_OF_BOUNDS ", 1, True),
    ("invalid | step=-1 | reason=out_of_bounds", 3, False),
    ("invalid | step=1.5 | reason=out_of_bounds", 3, False),
    ("invalid | step=1 | reason=collision", 3, False),
    ("valid | step=0 | reason=out_of_bounds", 3, False),
    ("invalid | step=" + "9" * 5000 + " | reason=out_of_bounds", 3, False),
])
def test_contextual_step_checks(text, length, accepted):
    result = parse_verification(text, plan_length=length)
    assert result["parse_success"] is accepted
    if not accepted:
        assert result["valid"] is None
        assert result["first_error_step"] is None
        assert result["reason"] is None
    assert result["raw_output"] == text


@pytest.mark.parametrize("value", [None, 42, [], {}, "", " \n\t"])
def test_empty_or_nontext_outputs_fail_explicitly(value):
    assert parse_plan(value)["parse_success"] is False
    assert parse_verification(value)["parse_success"] is False


def test_action_whitespace_and_unknown_token():
    assert parse_plan(" LEFT,\n teleport, down ")["plan"] == ["left", "teleport", "down"]


def test_inconsistent_executor_label_is_rejected():
    row = deepcopy(samples()[4])
    row["executor_label"]["success"] = True
    with pytest.raises(ValueError):
        convert_sample(row)


def test_step_number_is_not_reinterpreted_as_ground_truth():
    # A wrong model diagnosis within bounds remains a prediction, not corrected.
    result = parse_verification("invalid | step=0 | reason=out_of_bounds", plan_length=3)
    assert result["parse_success"] is True
    assert result["first_error_step"] == 0
