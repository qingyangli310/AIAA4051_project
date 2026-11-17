from src.smoke_integration import (
    DeterministicRepairPlaceholder,
    ExecutorVerifierAdapter,
    FixedPlanGenerator,
    RealExecutorAdapter,
)
from src.pipeline import run_pipeline


GRID = {
    "environment_id": "smoke-grid",
    "rows": 3,
    "cols": 3,
    "start": [0, 0],
    "goal": [2, 2],
    "obstacles": [],
}


def test_executor_adapter_routes_feasible_incomplete_plan_to_repair() -> None:
    result = ExecutorVerifierAdapter().verify(GRID, ["down"])
    assert result["executor_valid"] is True
    assert result["success"] is False
    assert result["valid"] is False
    assert result["reason"] == "goal_not_reached"


def test_real_executor_and_placeholder_complete_pipeline() -> None:
    result = run_pipeline(
        GRID,
        FixedPlanGenerator(["down"]),
        ExecutorVerifierAdapter(),
        DeterministicRepairPlaceholder(["down", "down", "right", "right"]),
        RealExecutorAdapter(),
        feedback_source="executor_feedback",
    )
    assert result["repair"]["prediction_kind"] == "deterministic_mock"
    assert result["final_result"]["success"] is True


def test_no_feedback_placeholder_preserves_candidate_and_failure() -> None:
    result = run_pipeline(
        GRID,
        FixedPlanGenerator(["down"]),
        ExecutorVerifierAdapter(),
        DeterministicRepairPlaceholder(
            ["down", "down", "right", "right"], keep_candidate=True
        ),
        RealExecutorAdapter(),
        feedback_source="no_feedback",
    )
    assert result["repair"]["feedback_source"] == "no_feedback"
    assert result["repaired_plan"] == ["down"]
    assert result["final_result"]["success"] is False
