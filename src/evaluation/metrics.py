"""Dependency-free implementations of the frozen Project 6 metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


INFEASIBLE_REASONS = {
    "obstacle_collision",
    "out_of_bounds",
    "invalid_action_token",
}


def safe_rate(numerator: int, denominator: int) -> float | None:
    """Return a rate, or ``None`` when the metric has no eligible examples."""
    return numerator / denominator if denominator else None


def is_feasible(record: dict[str, Any]) -> bool:
    """Apply the frozen feasibility rule to one flattened result record."""
    if record.get("final_parse_success") is False:
        return False
    if record.get("repair_parse_success") is False:
        return False
    if (
        record.get("candidate_parse_success") is False
        and record.get("repair_parse_success") is not True
    ):
        return False
    if record.get("executor_error") or record.get("status") == "error":
        return False
    reason = record.get("final_reason")
    return reason not in INFEASIBLE_REASONS


def optimality_score(
    *, successful: bool, optimal_length: int | None, predicted_length: int | None
) -> float | None:
    """Compute optimal_length / predicted_length only for successful plans."""
    if not successful:
        return None
    if not isinstance(optimal_length, int) or optimal_length < 0:
        raise ValueError("successful rows require a non-negative optimal_length")
    if not isinstance(predicted_length, int) or predicted_length <= 0:
        if optimal_length == predicted_length == 0:
            return 1.0
        raise ValueError("successful rows require a positive predicted_length")
    if optimal_length > predicted_length:
        raise ValueError("predicted_length cannot beat a verified optimal_length")
    return optimal_length / predicted_length


def verification_macro_f1(
    predicted_valid: Sequence[bool | None], ground_truth_valid: Sequence[bool]
) -> float | None:
    """Macro-F1 over the fixed labels valid=True and invalid=False.

    ``None`` predictions represent parser failures and count as false negatives
    for the true class without becoming a third label.
    """
    if len(predicted_valid) != len(ground_truth_valid):
        raise ValueError("prediction and ground-truth lengths differ")
    if not ground_truth_valid:
        return None

    scores: list[float] = []
    for label in (False, True):
        tp = sum(p is label and g is label for p, g in zip(predicted_valid, ground_truth_valid))
        fp = sum(p is label and g is not label for p, g in zip(predicted_valid, ground_truth_valid))
        fn = sum(p is not label and g is label for p, g in zip(predicted_valid, ground_truth_valid))
        denominator = 2 * tp + fp + fn
        scores.append((2 * tp / denominator) if denominator else 0.0)
    return sum(scores) / len(scores)


def first_error_accuracy(
    predicted_steps: Sequence[int | None], ground_truth_steps: Sequence[int | None]
) -> float | None:
    """Exact step accuracy on rows whose ground truth contains an error."""
    if len(predicted_steps) != len(ground_truth_steps):
        raise ValueError("prediction and ground-truth lengths differ")
    eligible = [
        (predicted, truth)
        for predicted, truth in zip(predicted_steps, ground_truth_steps)
        if truth is not None
    ]
    return safe_rate(sum(predicted == truth for predicted, truth in eligible), len(eligible))


def evaluate_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compute all frozen metrics plus transparent numerator/denominator counts."""
    total = len(records)
    successes = sum(record.get("final_success") is True for record in records)
    feasible = sum(is_feasible(record) for record in records)

    repair_rows = [record for record in records if record.get("candidate_valid") is False]
    repair_successes = sum(record.get("repair_success") is True for record in repair_rows)

    verification_rows = [
        record
        for record in records
        if isinstance(record.get("ground_truth_valid"), bool)
    ]
    macro_f1 = verification_macro_f1(
        [record.get("predicted_valid") for record in verification_rows],
        [record["ground_truth_valid"] for record in verification_rows],
    )
    step_accuracy = first_error_accuracy(
        [record.get("predicted_error_step") for record in verification_rows],
        [record.get("ground_truth_error_step") for record in verification_rows],
    )

    optimality_values = [
        optimality_score(
            successful=record.get("final_success") is True,
            optimal_length=record.get("optimal_length"),
            predicted_length=record.get("predicted_length"),
        )
        for record in records
    ]
    scored_optimality = [value for value in optimality_values if value is not None]

    return {
        "examples": total,
        "success_rate": safe_rate(successes, total),
        "feasibility": safe_rate(feasible, total),
        "verification_macro_f1": macro_f1,
        "first_error_accuracy": step_accuracy,
        "repair_success_rate": safe_rate(repair_successes, len(repair_rows)),
        "end_to_end_success_rate": safe_rate(successes, total),
        "mean_optimality": (
            sum(scored_optimality) / len(scored_optimality)
            if scored_optimality
            else None
        ),
        "counts": {
            "success": [successes, total],
            "feasible": [feasible, total],
            "repair_success": [repair_successes, len(repair_rows)],
            "optimality_scored": len(scored_optimality),
            "verification_scored": len(verification_rows),
        },
    }
