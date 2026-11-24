"""Frozen text-to-text prompts shared by Part 1 training and inference."""

from __future__ import annotations

from typing import Any


PROTOCOL_VERSION = "project6-part1-v2"


def validate_grid(grid: dict[str, Any]) -> None:
    required = {"environment_id", "rows", "cols", "start", "goal", "obstacles"}
    if not isinstance(grid, dict) or required - grid.keys():
        raise ValueError(f"grid requires fields: {sorted(required)}")
    if not isinstance(grid["environment_id"], str) or not grid["environment_id"]:
        raise ValueError("environment_id must be a non-empty string")
    if type(grid["rows"]) is not int or type(grid["cols"]) is not int:
        raise ValueError("rows and cols must be integers")
    if grid["rows"] <= 0 or grid["cols"] <= 0:
        raise ValueError("rows and cols must be positive")
    if not isinstance(grid["obstacles"], list):
        raise ValueError("obstacles must be a list")
    cells = [grid["start"], grid["goal"], *grid["obstacles"]]
    for cell in cells:
        if not isinstance(cell, list) or len(cell) != 2 or any(type(x) is not int for x in cell):
            raise ValueError("coordinates must be [row, col] integer lists")
        if not (0 <= cell[0] < grid["rows"] and 0 <= cell[1] < grid["cols"]):
            raise ValueError("grid coordinate is out of bounds")
    if grid["start"] in grid["obstacles"] or grid["goal"] in grid["obstacles"]:
        raise ValueError("start and goal cannot be obstacles")


def render_ascii_grid(grid: dict[str, Any]) -> str:
    """Render the spatial state explicitly for text-to-text models."""
    validate_grid(grid)
    start = tuple(grid["start"])
    goal = tuple(grid["goal"])
    obstacles = {tuple(cell) for cell in grid["obstacles"]}
    lines = []
    for row in range(grid["rows"]):
        cells = []
        for col in range(grid["cols"]):
            position = (row, col)
            if position == start:
                cells.append("S")
            elif position == goal:
                cells.append("G")
            elif position in obstacles:
                cells.append("#")
            else:
                cells.append(".")
        lines.append(" ".join(cells))
    return "\n".join(lines)


def grid_text(grid: dict[str, Any]) -> str:
    validate_grid(grid)
    delta_row = grid["goal"][0] - grid["start"][0]
    delta_col = grid["goal"][1] - grid["start"][1]
    return (
        f"Grid size: {grid['rows']} rows and {grid['cols']} columns.\n"
        f"Start: {tuple(grid['start'])}.\n"
        f"Goal: {tuple(grid['goal'])}.\n"
        f"Goal relative to start: delta_row={delta_row:+d}, delta_col={delta_col:+d}.\n"
        f"Obstacles: {[tuple(cell) for cell in grid['obstacles']]}.\n"
        "Grid map (S=start, G=goal, #=obstacle, .=free):\n"
        f"{render_ascii_grid(grid)}\n"
        "Actions: up decreases row; down increases row; left decreases column; "
        "right increases column.\n"
    )


def build_generation_prompt(grid: dict[str, Any]) -> str:
    return (
        "Task: generate a shortest valid path for the grid.\n"
        "Coordinates use zero-based (row, column).\n"
        + grid_text(grid)
        + "Stay in bounds, avoid obstacles, and finish at the goal.\n"
        "Output only comma-separated actions from: up, down, left, right.\n"
        "Path:"
    )


def build_verification_prompt(grid: dict[str, Any], plan: list[str]) -> str:
    if not isinstance(plan, list) or any(not isinstance(action, str) for action in plan):
        raise ValueError("plan must be a list of strings")
    return (
        "Task: verify the candidate path.\n"
        "Coordinates and action steps are zero-based.\n"
        + grid_text(grid)
        + f"Candidate path: {', '.join(plan)}\n"
        f"If all {len(plan)} actions finish before the goal, use step={len(plan)}.\n"
        "Output exactly valid or invalid | step=<integer> | reason=<error_type>.\n"
        "Allowed reasons: obstacle_collision, out_of_bounds, invalid_action_token, "
        "goal_not_reached.\n"
        "Answer:"
    )


def build_repair_prompt(
    grid: dict[str, Any],
    candidate_plan: list[str],
    feedback_source: str,
    verifier_feedback: dict[str, Any] | None,
) -> str:
    if feedback_source not in {"no_feedback", "executor_feedback", "self_feedback"}:
        raise ValueError("unsupported feedback source")
    if not isinstance(candidate_plan, list):
        raise ValueError("candidate_plan must be a list")
    if feedback_source == "no_feedback":
        feedback = "none"
    else:
        if not isinstance(verifier_feedback, dict):
            raise ValueError("feedback-enabled repair requires verifier feedback")
        feedback = (
            f"invalid | step={verifier_feedback['first_error_step']} | "
            f"reason={verifier_feedback['reason']}"
        )
    return (
        "Task: repair the candidate path so that it reaches the goal.\n"
        "Coordinates and action steps are zero-based.\n"
        + grid_text(grid)
        + f"Candidate path: {', '.join(candidate_plan)}\n"
        f"Feedback source: {feedback_source}\n"
        f"Verifier feedback: {feedback}\n"
        "Output only the complete comma-separated repaired path using: "
        "up, down, left, right.\n"
        "Repaired path:"
    )


def format_repair_trace_prompt(prompt: str) -> str:
    """Use the Trace instructions shared by Repair training and inference."""
    marker = "Output only the complete comma-separated repaired path using: "
    if marker not in prompt:
        raise ValueError("Repair prompt marker missing")
    prefix = prompt.split(marker, 1)[0]
    return (
        prefix
        + "Output exactly one Trace: line with the start coordinate, then each action "
        "and its resulting coordinate. Use coordinates formatted (row,col). Stop at "
        "the goal.\nAnswer:"
    )
