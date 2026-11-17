"""Adapters used by the November 15-16 deterministic integration smoke test.

These components deliberately do not pretend to be trained models.  They replay
frozen candidate plans and provide a deterministic Repair placeholder while the
real Generator/Verifier/Repair checkpoints are unavailable.
"""

from __future__ import annotations

from typing import Any

from src.executor.executor import execute_plan


class FixedPlanGenerator:
    """Replay one frozen candidate plan as an explicitly simulated prediction."""

    def __init__(self, plan: list[str]) -> None:
        self.plan = list(plan)

    def generate(self, grid: dict[str, Any]) -> dict[str, Any]:
        del grid
        return {
            "raw_output": ", ".join(self.plan),
            "plan": list(self.plan),
            "parse_success": True,
            "prediction_kind": "fixed_data_replay",
        }


class ExecutorVerifierAdapter:
    """Expose A's real Executor through C's Verification-task contract.

    A uses ``valid=True, success=False`` for a feasible but incomplete plan.
    Repair routing needs task validity (the plan solves the task), so the
    adapter maps ``success`` to ``valid`` while preserving Executor validity.
    """

    def verify(self, grid: dict[str, Any], plan: list[str]) -> dict[str, Any]:
        result = execute_plan(grid, plan)
        return {
            "valid": result["success"],
            "executor_valid": result["valid"],
            "success": result["success"],
            "first_error_step": result["first_error_step"],
            "reason": result["reason"],
            "prediction_kind": "executor_ground_truth_adapter",
        }


class DeterministicRepairPlaceholder:
    """Deterministic branch placeholder for pipeline and metric validation."""

    def __init__(
        self,
        reference_plan: list[str],
        *,
        keep_candidate: bool = False,
        inject_failure: bool = False,
    ) -> None:
        self.reference_plan = list(reference_plan)
        self.keep_candidate = keep_candidate
        self.inject_failure = inject_failure

    def repair(
        self,
        environment_id: str,
        grid: dict[str, Any],
        candidate_plan: list[str],
        feedback_source: str,
        verifier_feedback: dict[str, Any] | None,
    ) -> dict[str, Any]:
        del environment_id, grid, verifier_feedback
        if self.keep_candidate or self.inject_failure:
            plan = list(candidate_plan)
        else:
            plan = list(self.reference_plan)
        return {
            "raw_output": ", ".join(plan),
            "repaired_plan": plan,
            "parse_success": True,
            "prediction_kind": "deterministic_mock",
            "feedback_source": feedback_source,
            "failure_injected": self.inject_failure,
        }


class RealExecutorAdapter:
    """Expose A's canonical Executor through C's pipeline interface."""

    def execute(self, grid: dict[str, Any], plan: list[str]) -> dict[str, Any]:
        return execute_plan(grid, plan)
