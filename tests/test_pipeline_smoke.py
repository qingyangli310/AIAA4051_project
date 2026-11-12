from src.mock_components import (
    MockExecutor,
    MockGenerator,
    MockRepairer,
    MockVerifier,
    execute_plan,
)
from src.pipeline import run_pipeline, run_pipeline_batch


GRID = {
    "environment_id": "grid_0001",
    "rows": 3,
    "cols": 3,
    "start": [0, 0],
    "goal": [2, 2],
    "obstacles": [[0, 2]],
}


class ValidGenerator:
    def generate(self, grid):
        del grid
        plan = ["down", "right", "down", "right"]
        return {"raw_output": ", ".join(plan), "plan": plan, "parse_success": True}


class FailingRepairer:
    def repair(self, **request):
        del request
        plan = ["down"]
        return {"raw_output": "down", "repaired_plan": plan, "parse_success": True}


class UnparseableRepairer:
    def repair(self, **request):
        del request
        return {"raw_output": "no path", "repaired_plan": [], "parse_success": False}


class RepairMustNotRun:
    def repair(self, **request):
        del request
        raise AssertionError("repair must not run for a valid candidate plan")


def test_mock_pipeline_repairs_collision() -> None:
    result = run_pipeline(
        GRID,
        MockGenerator(),
        MockVerifier(),
        MockRepairer(),
        MockExecutor(),
    )
    assert result["verification"] == {
        "valid": False,
        "first_error_step": 1,
        "reason": "obstacle_collision",
    }
    assert result["repaired_plan"] == ["down", "right", "down", "right"]
    assert result["final_result"]["success"] is True


def test_error_steps_are_zero_based() -> None:
    result = execute_plan(GRID, ["right", "right", "down"])
    assert result["first_error_step"] == 1


def test_goal_not_reached_uses_plan_length() -> None:
    plan = ["down"]
    result = execute_plan(GRID, plan)
    assert result["reason"] == "goal_not_reached"
    assert result["first_error_step"] == len(plan)


def test_valid_candidate_bypasses_repair() -> None:
    result = run_pipeline(
        GRID,
        ValidGenerator(),
        MockVerifier(),
        RepairMustNotRun(),
        MockExecutor(),
    )
    assert result["verification"]["valid"] is True
    assert result["repair"] is None
    assert result["final_result"]["success"] is True


def test_out_of_bounds_uses_zero_based_step() -> None:
    result = execute_plan(GRID, ["up"])
    assert result["reason"] == "out_of_bounds"
    assert result["first_error_step"] == 0


def test_invalid_action_token_uses_zero_based_step() -> None:
    result = execute_plan(GRID, ["down", "jump"])
    assert result["reason"] == "invalid_action_token"
    assert result["first_error_step"] == 1


def test_failed_repair_is_preserved_in_final_result() -> None:
    result = run_pipeline(
        GRID,
        MockGenerator(),
        MockVerifier(),
        FailingRepairer(),
        MockExecutor(),
    )
    assert result["repair"]["parse_success"] is True
    assert result["final_result"]["success"] is False
    assert result["final_result"]["reason"] == "goal_not_reached"


def test_unparseable_repair_counts_as_final_failure() -> None:
    result = run_pipeline(
        GRID,
        MockGenerator(),
        MockVerifier(),
        UnparseableRepairer(),
        MockExecutor(),
    )
    assert result["repair"]["parse_success"] is False
    assert result["final_result"]["success"] is False


class CapturingRepairer:
    def __init__(self):
        self.request = None

    def repair(self, **request):
        self.request = request
        plan = ["down", "right", "down", "right"]
        return {"raw_output": ", ".join(plan), "repaired_plan": plan, "parse_success": True}


def test_no_feedback_hides_verifier_content_from_repairer() -> None:
    repairer = CapturingRepairer()
    result = run_pipeline(
        GRID,
        MockGenerator(),
        MockVerifier(),
        repairer,
        MockExecutor(),
        feedback_source="no_feedback",
    )
    assert result["feedback_source"] == "no_feedback"
    assert repairer.request["feedback_source"] == "no_feedback"
    assert repairer.request["verifier_feedback"] is None


def test_batch_preserves_error_and_continues() -> None:
    broken_grid = {"environment_id": "broken"}
    records = run_pipeline_batch(
        [broken_grid, GRID],
        MockGenerator(),
        MockVerifier(),
        MockRepairer(),
        MockExecutor(),
    )
    assert records[0]["status"] == "error"
    assert records[0]["environment_id"] == "broken"
    assert records[1]["status"] == "ok"
    assert records[1]["final_result"]["success"] is True
