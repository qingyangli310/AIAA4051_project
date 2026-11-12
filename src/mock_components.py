"""Deterministic mock components for the first integration milestone."""

from __future__ import annotations

from typing import Any

from src.repair_schema import VALID_ACTIONS


MOVE_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def execute_plan(grid: dict[str, Any], plan: list[str]) -> dict[str, Any]:
    """Execute a plan with 0-based error steps and len(plan) for goal failure."""
    position = tuple(grid["start"])
    obstacles = {tuple(item) for item in grid["obstacles"]}
    trace: list[dict[str, Any]] = []

    for step, action in enumerate(plan):
        if action not in VALID_ACTIONS:
            return _failure(position, trace, step, "invalid_action_token")

        row_delta, col_delta = MOVE_DELTAS[action]
        next_position = (position[0] + row_delta, position[1] + col_delta)
        in_bounds = (
            0 <= next_position[0] < grid["rows"]
            and 0 <= next_position[1] < grid["cols"]
        )
        if not in_bounds:
            return _failure(position, trace, step, "out_of_bounds")
        if next_position in obstacles:
            return _failure(position, trace, step, "obstacle_collision")

        position = next_position
        trace.append(
            {
                "step": step,
                "action": action,
                "position": list(position),
                "status": "ok",
            }
        )

    if list(position) != grid["goal"]:
        return _failure(position, trace, len(plan), "goal_not_reached")

    return {
        "valid": True,
        "success": True,
        "first_error_step": None,
        "reason": None,
        "final_position": list(position),
        "executed_steps": len(plan),
        "trace": trace,
    }


def _failure(
    position: tuple[int, int],
    trace: list[dict[str, Any]],
    step: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "valid": False,
        "success": False,
        "first_error_step": step,
        "reason": reason,
        "final_position": list(position),
        "executed_steps": len(trace),
        "trace": trace,
    }


class MockGenerator:
    def generate(self, grid: dict[str, Any]) -> dict[str, Any]:
        del grid
        plan = ["right", "right", "down"]
        return {"raw_output": ", ".join(plan), "plan": plan, "parse_success": True}


class MockVerifier:
    def verify(self, grid: dict[str, Any], plan: list[str]) -> dict[str, Any]:
        result = execute_plan(grid, plan)
        return {
            "valid": result["valid"],
            "first_error_step": result["first_error_step"],
            "reason": result["reason"],
        }


class MockRepairer:
    def repair(
        self,
        environment_id: str,
        grid: dict[str, Any],
        candidate_plan: list[str],
        feedback_source: str,
        verifier_feedback: dict[str, Any] | None,
    ) -> dict[str, Any]:
        del environment_id, grid, candidate_plan, feedback_source, verifier_feedback
        plan = ["down", "right", "down", "right"]
        return {
            "raw_output": ", ".join(plan),
            "repaired_plan": plan,
            "parse_success": True,
        }


class MockExecutor:
    def execute(self, grid: dict[str, Any], plan: list[str]) -> dict[str, Any]:
        return execute_plan(grid, plan)
