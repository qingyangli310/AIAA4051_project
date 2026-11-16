"""Keep A Executor feasibility and C Mock Verifier semantics distinct."""
from src.executor.executor import execute_plan
from src.mock_components import MockVerifier


def test_goal_not_reached_semantics_are_explicit_across_a_and_c_interfaces():
    grid = {"environment_id": "compat-grid", "rows": 2, "cols": 2,
            "start": [0, 0], "goal": [1, 1], "obstacles": []}
    plan = ["right"]

    executor = execute_plan(grid, plan)
    verifier = MockVerifier().verify(grid, plan)

    assert executor["valid"] is True
    assert executor["success"] is False
    assert executor["reason"] == "goal_not_reached"
    assert executor["first_error_step"] == len(plan)
    # C's current Verification-shaped Mock treats failure to reach the goal as
    # invalid for its control flow. Preserve, document, and do not conflate it.
    assert verifier == {"valid": False, "first_error_step": len(plan), "reason": "goal_not_reached"}
