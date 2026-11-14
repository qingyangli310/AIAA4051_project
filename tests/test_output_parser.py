"""Parser contract tests; no model loading or network access is needed.

Run manually from the project root:
    python -m pytest tests/test_output_parser.py -v
"""

import pytest

from src.models.output_parser import parse_plan


@pytest.mark.parametrize(
    "raw_output, expected_plan",
    [
        pytest.param(
            "right, down, down, right",
            ["right", "down", "down", "right"],
            id="preserves-action-order-and-repetition",
        ),
        pytest.param(
            " RIGHT , Down ",
            ["right", "down"],
            id="normalizes-case-and-whitespace",
        ),
        pytest.param("up", ["up"], id="accepts-single-action"),
        pytest.param(
            "right, jump, down",
            ["right", "jump", "down"],
            id="preserves-invalid-action-for-executor",
        ),
        pytest.param("up, move_up, down", ["up", "move_up", "down"], id="unknown-underscore"),
        pytest.param("up, turn-left, down", ["up", "turn-left", "down"], id="unknown-hyphen"),
        pytest.param("up, up2, down", ["up", "up2", "down"], id="unknown-alphanumeric"),
        pytest.param("up, 123, down", ["up", "123", "down"], id="numeric-token-kept-as-string"),
        pytest.param("up, 左移, down", ["up", "左移", "down"], id="unknown-unicode"),
        pytest.param("up, ?, down", ["up", "?", "down"], id="unknown-symbol"),
        pytest.param("up, down.", ["up", "down."], id="does-not-strip-punctuation"),
        pytest.param(" MOVE_UP ,\n Down ", ["move_up", "down"], id="normalizes-only-case-and-edges"),
    ],
)
def test_parse_plan_accepts_action_sequence(raw_output, expected_plan):
    result = parse_plan(raw_output)

    assert result["parse_success"] is True
    assert result["plan"] == expected_plan
    # Keep the original text for auditing, including case and whitespace.
    assert result["raw_output"] == raw_output


@pytest.mark.parametrize(
    "raw_output",
    [
        pytest.param("", id="empty-output"),
        pytest.param(" \n\t ", id="whitespace-only-output"),
        pytest.param("right,,down", id="empty-action-between-commas"),
        pytest.param(",right", id="leading-comma"),
        pytest.param("right,", id="trailing-comma"),
        pytest.param("right down", id="missing-comma"),
        pytest.param("The path is right, down.", id="explanatory-sentence"),
        pytest.param("[(0, 0), (1, 0)]", id="coordinates-instead-of-actions"),
        pytest.param('["up", "down"]', id="json-array-not-comma-text"),
        pytest.param("up, move up, down", id="internal-space-not-single-token"),
        pytest.param("up, move\nup, down", id="internal-newline-not-single-token"),
        pytest.param("```up, down```", id="markdown-wrapper"),
        pytest.param("up, \x00, down", id="control-character"),
        pytest.param("up, move_up,", id="no-partial-salvage-with-empty-field"),
        pytest.param(
            "by 1; down increases row by 1. left decreases col by 1; "
            "right increases col by 1. Avoid obstacles and stay inside "
            "the grid. Output only actions separated by commas: up, "
            "down, left, right. Path: [(0, 0)].",
            id="actual-t5-output-is-not-an-action-sequence",
        ),
    ],
)
def test_parse_plan_rejects_malformed_output(raw_output):
    result = parse_plan(raw_output)

    assert result["parse_success"] is False
    # Do not salvage a partial plan by extracting known action words.
    assert result["plan"] == []
    assert result["raw_output"] == raw_output
