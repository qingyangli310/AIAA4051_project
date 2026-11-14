"""Offline interface checks with fixed text, not model accuracy tests."""

import pytest

from src.models.api import GridModel
from src.models.prompts import build_verification_prompt
from src.models.backend import output_length_metadata


GRID = {
    "environment_id": "toy_grid_0001",
    "rows": 3, "cols": 3, "start": [0, 0], "goal": [2, 2],
    "obstacles": [[0, 2]],
}


class FixedTextBackend:
    """Replace only neural inference; run real prompts and parsers."""

    def __init__(self, text):
        self.text = text

    def generate_text(self, prompt):
        return {"raw_output": self.text, "metadata": {"backend": "test_fixture"}}


def test_generation_returns_protocol_fields_and_audit_metadata():
    result = GridModel(backend=FixedTextBackend(" RIGHT, jump, down ")).generate_plan(GRID)
    assert result["plan"] == ["right", "jump", "down"]
    assert result["parse_success"] is True
    assert result["raw_output"] == " RIGHT, jump, down "
    assert result["metadata"]["environment_id"] == "toy_grid_0001"
    assert result["metadata"]["step_index_base"] == 0


def test_unknown_action_survives_generation_and_verification_handoff():
    raw = " RIGHT, MOVE_UP, down "
    generation = GridModel(backend=FixedTextBackend(raw)).generate_plan(GRID)
    assert generation["parse_success"] is True
    assert generation["plan"] == ["right", "move_up", "down"]
    assert generation["raw_output"] == raw
    # Mock diagnosis tests only the B interface, not A's actual executor.
    verification = GridModel(backend=FixedTextBackend(
        "invalid | step=1 | reason=invalid_action_token"
    )).verify_plan(GRID, generation["plan"])
    assert "Candidate path: right, move_up, down\n" in verification["metadata"]["prompt"]
    assert verification["parse_success"] is True
    assert verification["valid"] is False
    assert verification["first_error_step"] == 1
    assert verification["reason"] == "invalid_action_token"


def test_verification_preserves_zero_based_prediction():
    result = GridModel(backend=FixedTextBackend(
        "invalid | step=0 | reason=out_of_bounds"
    )).verify_plan(GRID, ["up"])
    assert result["valid"] is False
    assert result["first_error_step"] == 0
    assert result["parse_success"] is True


def test_second_action_collision_uses_step_one():
    result = GridModel(backend=FixedTextBackend(
        "invalid | step=1 | reason=obstacle_collision"
    )).verify_plan(GRID, ["right", "right", "down"])
    assert result["first_error_step"] == 1
    assert result["reason"] == "obstacle_collision"


def test_parse_failure_stays_unknown_and_keeps_raw_output():
    result = GridModel(backend=FixedTextBackend("The path looks fine.")).verify_plan(GRID, [])
    assert result["parse_success"] is False
    assert result["valid"] is None
    assert result["raw_output"] == "The path looks fine."


def test_prompt_states_zero_based_steps_and_actual_end_checkpoint():
    prompt = build_verification_prompt(GRID, ["right", "down"])
    assert "Action steps start at 0" in prompt
    assert "step=2 for goal_not_reached" in prompt
    assert "step=0 for goal_not_reached" in build_verification_prompt(GRID, [])


def test_empty_candidate_can_report_goal_not_reached_at_zero():
    result = GridModel(backend=FixedTextBackend(
        "invalid | step=0 | reason=goal_not_reached"
    )).verify_plan(GRID, [])
    assert result["parse_success"] is True
    assert result["first_error_step"] == 0


def test_unknown_action_is_passed_to_verifier_without_repair():
    prompt = build_verification_prompt(GRID, ["right", "jump"])
    assert "Candidate path: right, jump" in prompt


def test_missing_environment_id_is_not_silently_invented():
    grid = {key: value for key, value in GRID.items() if key != "environment_id"}
    with pytest.raises(ValueError, match="environment_id"):
        GridModel(backend=FixedTextBackend("right")).generate_plan(grid)


def test_candidate_must_be_a_list_not_raw_text():
    with pytest.raises(ValueError, match="list"):
        GridModel(backend=FixedTextBackend("valid")).verify_plan(GRID, "right, down")


def test_inference_error_is_not_converted_to_a_negative_prediction():
    class FailingBackend:
        def generate_text(self, prompt):
            raise RuntimeError("inference unavailable")

    with pytest.raises(RuntimeError, match="inference unavailable"):
        GridModel(backend=FailingBackend()).verify_plan(GRID, ["right"])


def test_eos_at_limit_still_records_reaching_limit():
    result = output_length_metadata(64, 2, 2, 64)
    assert result["ended_with_eos"] is True
    assert result["reached_output_limit"] is True


def test_eos_before_limit_does_not_record_reaching_limit():
    result = output_length_metadata(12, 1, 1, 64)
    assert result["ended_with_eos"] is True
    assert result["reached_output_limit"] is False
