"""Parse model output without repairing or filtering candidate actions."""

import re


VALID_ACTIONS = {"up", "down", "left", "right"}

# Reserved syntax for arrays, coordinates, quoted values and Markdown code.
# A bare action token can contain other printable, non-whitespace characters.
# Whether it is one of VALID_ACTIONS is exclusively an executor decision.
PLAN_WRAPPER_CHARS = frozenset("[]{}()\"'`")


def _is_bare_plan_token(token: str) -> bool:
    return bool(token) and all(
        character.isprintable()
        and not character.isspace()
        and character not in PLAN_WRAPPER_CHARS
        for character in token
    )

ERROR_TYPES = {
    "obstacle_collision",
    "out_of_bounds",
    "invalid_action_token",
    "goal_not_reached",
}


def parse_plan(raw_output: str) -> dict:
    """Parse comma-separated bare tokens; success does not imply a valid path.

    Unknown tokens such as ``move_up``, ``turn-left``, ``123`` and ``?``
    are retained in order for the executor to check. Only case and surrounding
    whitespace are normalized; punctuation is never removed to repair actions.
    Empty fields, internal whitespace, nonprintable characters and wrappers
    (arrays, coordinates, quoted strings, code fences) fail the whole parse.
    The raw output is always kept unchanged, including on format failure.
    """
    result = {
        "raw_output": raw_output,
        "plan": [],
        "parse_success": False,
    }

    if not isinstance(raw_output, str):
        return result
    text = raw_output.strip().lower()
    if not text:
        return result

    tokens = [token.strip() for token in text.split(",")]
    if not all(_is_bare_plan_token(token) for token in tokens):
        return result

    result["plan"] = tokens
    result["parse_success"] = True
    return result


def parse_verification(raw_output: str, plan_length=None) -> dict:
    """Parse a model judgment, without checking whether it is correct.

    Accept ``valid`` or ``invalid | step=N | reason=ERROR_TYPE``.
    Case and whitespace are normalized, but field order is fixed.
    On parse failure, ``valid`` is None, not False. Consumers must check
    ``parse_success`` before interpreting the prediction.

    Steps are nonnegative, 0-based integers. Checking a step against the
    candidate length is enabled by passing plan_length. This checks bounds,
    not whether a predicted error actually occurred. Ground truth stays with A.
    """
    result = {
        "raw_output": raw_output,
        "parse_success": False,
        "valid": None,
        "first_error_step": None,
        "reason": None,
    }
    if plan_length is not None and (type(plan_length) is not int or plan_length < 0):
        raise ValueError("plan_length must be a nonnegative integer")
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

    try:
        step = int(match.group(1))
    except ValueError:
        return result
    reason = match.group(2)
    if step < 0 or reason not in ERROR_TYPES:
        return result
    if plan_length is not None:
        if reason == "goal_not_reached":
            if step != plan_length:
                return result
        elif step >= plan_length:
            return result

    result.update(
        parse_success=True,
        valid=False,
        first_error_step=step,
        reason=reason,
    )
    return result
