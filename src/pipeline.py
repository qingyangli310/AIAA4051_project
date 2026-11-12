"""End-to-end Generate-Verify-Repair-Execute orchestration."""

from __future__ import annotations

from typing import Any

from src.repair_schema import build_repair_request, validate_repair_output


def run_pipeline(
    grid: dict[str, Any],
    generator: Any,
    verifier: Any,
    repairer: Any,
    executor: Any,
    *,
    feedback_source: str = "self_feedback",
) -> dict[str, Any]:
    """Run one pipeline example and preserve every intermediate result."""
    generation = generator.generate(grid)
    candidate_plan = generation["plan"]
    verification = verifier.verify(grid=grid, plan=candidate_plan)

    repair = None
    if verification["valid"]:
        repaired_plan = candidate_plan
    else:
        repair_request = build_repair_request(
            grid,
            candidate_plan,
            verification,
            feedback_source=feedback_source,
        )
        repair = repairer.repair(**repair_request)
        validate_repair_output(repair)
        repaired_plan = repair["repaired_plan"]

    final_result = executor.execute(grid=grid, plan=repaired_plan)
    return {
        "environment_id": grid["environment_id"],
        "feedback_source": feedback_source,
        "generation": generation,
        "candidate_plan": candidate_plan,
        "verification": verification,
        "repair": repair,
        "repaired_plan": repaired_plan,
        "final_result": final_result,
    }


def run_pipeline_batch(
    grids: list[dict[str, Any]],
    generator: Any,
    verifier: Any,
    repairer: Any,
    executor: Any,
    *,
    feedback_source: str = "self_feedback",
) -> list[dict[str, Any]]:
    """Run a batch while preserving per-sample failures instead of aborting."""
    records: list[dict[str, Any]] = []
    for grid in grids:
        try:
            result = run_pipeline(
                grid,
                generator,
                verifier,
                repairer,
                executor,
                feedback_source=feedback_source,
            )
            records.append({"status": "ok", **result})
        except Exception as exc:  # The caller receives an auditable failure record.
            records.append(
                {
                    "status": "error",
                    "environment_id": grid.get("environment_id"),
                    "feedback_source": feedback_source,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
    return records
