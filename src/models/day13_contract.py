"""Day-13 sample conversion. Standard library only; no model/executor loading."""

from src.models.output_parser import VALID_ACTIONS, parse_verification
from src.models.prompts import build_generation_prompt, build_verification_prompt, validate_grid

GRID_FIELDS = ("environment_id", "rows", "cols", "start", "goal", "obstacles")
LABEL_FIELDS = ("valid", "first_error_step", "reason")


def executor_to_verification(label, plan):
    """A.valid is movement feasibility; B.valid is A.success (task completion)."""
    if not isinstance(label, dict):
        raise ValueError("executor_label must be an object")
    for key in ("valid", "success"):
        if type(label.get(key)) is not bool:
            raise ValueError(f"executor_label.{key} must be boolean")
    if "reason" not in label or "first_error_step" not in label:
        raise ValueError("executor_label needs reason and first_error_step")
    if label["success"]:
        if not label["valid"] or label["reason"] is not None or label["first_error_step"] is not None:
            raise ValueError("inconsistent successful executor label")
        target = "valid"
    else:
        step, reason = label["first_error_step"], label["reason"]
        if type(step) is not int:
            raise ValueError("failure step must be an integer")
        if label["valid"] != (reason == "goal_not_reached"):
            raise ValueError("executor movement validity is inconsistent with reason")
        target = f"invalid | step={step} | reason={reason}"
    parsed = parse_verification(target, plan_length=len(plan))
    if not parsed["parse_success"]:
        raise ValueError("executor label has an unsupported reason or illegal step")
    return target, {key: parsed[key] for key in LABEL_FIELDS}


def convert_sample(row):
    """Return two audit/roundtrip records, NOT a formal train/dev assignment."""
    if not isinstance(row, dict):
        raise ValueError("sample must be an object")
    for field in ("sample_id", "source_file"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    if type(row.get("source_index")) is not int or row["source_index"] < 0:
        raise ValueError("source_index must be a nonnegative integer")
    grid = {key: row[key] for key in GRID_FIELDS}
    validate_grid(grid)
    correct = row["correct_plan"]
    if not isinstance(correct, list) or not correct or any(
        not isinstance(action, str) or action not in VALID_ACTIONS for action in correct
    ):
        raise ValueError("correct_plan must be a nonempty canonical action list")
    candidate = row["candidate_plan"]
    # Validate candidate type without dropping unknown tokens, e.g. teleport.
    verification_input = build_verification_prompt(grid, candidate)
    target, expected = executor_to_verification(row["executor_label"], candidate)
    shared = {
        "sample_id": row["sample_id"], "environment_id": row["environment_id"],
        "source_file": row["source_file"], "source_index": row["source_index"],
        "usage": "integration_only", "grid": grid,
    }
    generation = {
        **shared, "task": "generation", "input_text": build_generation_prompt(grid),
        "target_text": ", ".join(correct), "expected_plan": correct,
    }
    verification = {
        **shared, "task": "verification", "candidate_plan": candidate,
        "input_text": verification_input, "target_text": target,
        "expected_verification": expected, "executor_label": row["executor_label"],
    }
    return generation, verification
