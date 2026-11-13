import pytest

from src.evaluation.metrics import (
    evaluate_records,
    first_error_accuracy,
    is_feasible,
    optimality_score,
    verification_macro_f1,
)


def test_goal_not_reached_is_feasible_but_not_successful() -> None:
    record = {
        "final_success": False,
        "final_reason": "goal_not_reached",
        "repair_parse_success": True,
    }
    assert is_feasible(record) is True


@pytest.mark.parametrize(
    "reason", ["obstacle_collision", "out_of_bounds", "invalid_action_token"]
)
def test_movement_or_token_failure_is_not_feasible(reason: str) -> None:
    assert is_feasible({"final_reason": reason}) is False


def test_parse_and_executor_failures_are_not_feasible() -> None:
    assert is_feasible({"repair_parse_success": False}) is False
    assert is_feasible({"final_parse_success": False}) is False
    assert is_feasible({"candidate_parse_success": False}) is False
    assert is_feasible({"executor_error": "boom"}) is False


def test_optimality_only_scores_successful_paths() -> None:
    assert optimality_score(successful=False, optimal_length=4, predicted_length=6) is None
    assert optimality_score(successful=True, optimal_length=4, predicted_length=4) == 1.0
    assert optimality_score(successful=True, optimal_length=4, predicted_length=6) == pytest.approx(2 / 3)


def test_optimality_rejects_inconsistent_lengths() -> None:
    with pytest.raises(ValueError, match="cannot beat"):
        optimality_score(successful=True, optimal_length=5, predicted_length=4)


def test_verification_macro_f1_uses_fixed_valid_invalid_labels() -> None:
    assert verification_macro_f1([True, False], [True, False]) == 1.0
    assert verification_macro_f1([True, True], [True, False]) == pytest.approx(1 / 3)
    assert verification_macro_f1([None, False], [True, False]) == 0.5


def test_first_error_accuracy_ignores_valid_ground_truth_rows() -> None:
    assert first_error_accuracy([None, 1, 4], [None, 1, 3]) == 0.5
    assert first_error_accuracy([None], [None]) is None


def test_evaluate_records_returns_all_frozen_metrics_and_counts() -> None:
    records = [
        {
            "candidate_valid": False,
            "repair_success": True,
            "repair_parse_success": True,
            "final_success": True,
            "final_reason": None,
            "ground_truth_valid": False,
            "predicted_valid": False,
            "ground_truth_error_step": 1,
            "predicted_error_step": 1,
            "optimal_length": 4,
            "predicted_length": 4,
        },
        {
            "candidate_valid": False,
            "repair_success": False,
            "repair_parse_success": True,
            "final_success": False,
            "final_reason": "goal_not_reached",
            "ground_truth_valid": True,
            "predicted_valid": True,
            "ground_truth_error_step": None,
            "predicted_error_step": None,
            "optimal_length": 4,
            "predicted_length": 3,
        },
        {
            "candidate_valid": True,
            "repair_success": False,
            "repair_parse_success": False,
            "final_success": False,
            "final_reason": None,
            "optimal_length": 4,
            "predicted_length": 0,
        },
    ]
    metrics = evaluate_records(records)
    assert metrics["success_rate"] == pytest.approx(1 / 3)
    assert metrics["feasibility"] == pytest.approx(2 / 3)
    assert metrics["verification_macro_f1"] == 1.0
    assert metrics["first_error_accuracy"] == 1.0
    assert metrics["repair_success_rate"] == 0.5
    assert metrics["end_to_end_success_rate"] == pytest.approx(1 / 3)
    assert metrics["mean_optimality"] == 1.0
    assert metrics["counts"]["repair_success"] == [1, 2]
