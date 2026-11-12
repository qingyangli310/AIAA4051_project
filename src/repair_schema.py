"""Validation helpers for the frozen repair interface."""

from __future__ import annotations

from typing import Any


ERROR_TYPES = {
    "obstacle_collision",
    "out_of_bounds",
    "invalid_action_token",
    "goal_not_reached",
}
VALID_ACTIONS = {"up", "down", "left", "right"}
FEEDBACK_SOURCES = {"no_feedback", "self_feedback", "executor_feedback"}


def build_repair_request(
    grid: dict[str, Any],
    candidate_plan: list[str],
    verifier_feedback: dict[str, Any],
    feedback_source: str = "self_feedback",
) -> dict[str, Any]:
    """Build and validate a repair request using the shared field names."""
    required_grid_fields = {
        "environment_id",
        "rows",
        "cols",
        "start",
        "goal",
        "obstacles",
    }
    missing = required_grid_fields - grid.keys()
    if missing:
        raise ValueError(f"grid is missing required fields: {sorted(missing)}")

    if verifier_feedback.get("valid") is not False:
        raise ValueError("repair requests require invalid verifier feedback")

    error_step = verifier_feedback.get("first_error_step")
    if not isinstance(error_step, int) or error_step < 0:
        raise ValueError("first_error_step must be a non-negative 0-based integer")

    reason = verifier_feedback.get("reason")
    if reason not in ERROR_TYPES:
        raise ValueError(f"unsupported verifier reason: {reason!r}")
    if reason == "goal_not_reached" and error_step != len(candidate_plan):
        raise ValueError("goal_not_reached must use first_error_step=len(candidate_plan)")
    if feedback_source not in FEEDBACK_SOURCES:
        raise ValueError(f"unsupported feedback_source: {feedback_source!r}")

    return {
        "environment_id": grid["environment_id"],
        "grid": grid,
        "candidate_plan": list(candidate_plan),
        "feedback_source": feedback_source,
        "verifier_feedback": (
            None if feedback_source == "no_feedback" else dict(verifier_feedback)
        ),
    }


def validate_repair_output(output: dict[str, Any]) -> None:
    """Validate the structured output returned by a repair component."""
    required = {"raw_output", "repaired_plan", "parse_success"}
    missing = required - output.keys()
    if missing:
        raise ValueError(f"repair output is missing fields: {sorted(missing)}")
    if not isinstance(output["repaired_plan"], list):
        raise ValueError("repaired_plan must be a list")
    if output["parse_success"] and any(
        action not in VALID_ACTIONS for action in output["repaired_plan"]
    ):
        raise ValueError("a successfully parsed repaired_plan contains an invalid action")
