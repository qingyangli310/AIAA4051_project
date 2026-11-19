"""Strict parsers for Part 1 model outputs."""

from __future__ import annotations

import re
from typing import Any


VALID_ACTIONS = {"up", "down", "left", "right"}
ACTION_DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
ERROR_TYPES = {
    "obstacle_collision",
    "out_of_bounds",
    "invalid_action_token",
    "goal_not_reached",
}
PLAN_WRAPPER_CHARS = frozenset("[]{}()\"'`")
TRACE_COORD = r"\(([0-9]+),([0-9]+)\)"
TRACE_PATTERN = re.compile(
    TRACE_COORD + r"(?: (?:up|down|left|right) " + TRACE_COORD + r")+"
)


def parse_plan(raw_output: Any) -> dict[str, Any]:
    result = {"raw_output": raw_output, "plan": [], "parse_success": False}
    if not isinstance(raw_output, str) or not raw_output.strip():
        return result
    tokens = [token.strip().lower() for token in raw_output.strip().split(",")]
    if not all(
        token
        and all(
            char.isprintable()
            and not char.isspace()
            and char not in PLAN_WRAPPER_CHARS
            for char in token
        )
        for token in tokens
    ):
        return result
    result.update(plan=tokens, parse_success=True)
    return result


def parse_trace(raw_output: Any, grid: dict[str, Any]) -> dict[str, Any]:
    """Parse one coordinate/action Trace and verify its internal transitions."""
    result = {"raw_output": raw_output, "plan": [], "parse_success": False}
    if not isinstance(raw_output, str):
        return result
    match = re.fullmatch(r"Trace:\s*(.*?)", raw_output.strip(), flags=re.DOTALL)
    if match is None or TRACE_PATTERN.fullmatch(match.group(1)) is None:
        return result
    body = match.group(1)
    positions = [
        [int(row), int(col)] for row, col in re.findall(TRACE_COORD, body)
    ]
    actions = re.findall(r"\b(up|down|left|right)\b", body)
    consistent = all(
        [
            positions[index][0] + ACTION_DELTAS[action][0],
            positions[index][1] + ACTION_DELTAS[action][1],
        ]
        == positions[index + 1]
        for index, action in enumerate(actions)
    )
    if consistent and positions[0] == grid["start"]:
        result.update(plan=actions, parse_success=True, trace_positions=positions)
    return result


def parse_verification(raw_output: Any, plan_length: int | None = None) -> dict[str, Any]:
    result = {
        "raw_output": raw_output,
        "parse_success": False,
        "valid": None,
        "first_error_step": None,
        "reason": None,
    }
    if plan_length is not None and (type(plan_length) is not int or plan_length < 0):
        raise ValueError("plan_length must be a non-negative integer")
    if not isinstance(raw_output, str):
        return result
    text = raw_output.strip().lower()
    if text == "valid":
        result.update(parse_success=True, valid=True)
        return result
    match = re.fullmatch(
        r"invalid\s*\|\s*step\s*=\s*([0-9]+)\s*\|\s*reason\s*=\s*([a-z_]+)",
        text,
    )
    if match is None:
        return result
    step, reason = int(match.group(1)), match.group(2)
    if reason not in ERROR_TYPES:
        return result
    if plan_length is not None:
        if reason == "goal_not_reached" and step != plan_length:
            return result
        if reason != "goal_not_reached" and step >= plan_length:
            return result
    result.update(
        parse_success=True,
        valid=False,
        first_error_step=step,
        reason=reason,
    )
    return result
