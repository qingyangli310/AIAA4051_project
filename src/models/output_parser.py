"""Strict parsers for Part 1 model outputs."""

from __future__ import annotations

import re
from typing import Any


VALID_ACTIONS = {"up", "down", "left", "right"}
ERROR_TYPES = {
    "obstacle_collision",
    "out_of_bounds",
    "invalid_action_token",
    "goal_not_reached",
}
PLAN_WRAPPER_CHARS = frozenset("[]{}()\"'`")


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
