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
) -> dict[str, Any]:
    """Run one pipeline example and preserve every intermediate result."""
    generation = generator.generate(grid)
    candidate_plan = generation["plan"]
    verification = verifier.verify(grid=grid, plan=candidate_plan)

    repair = None
    if verification["valid"]:
        repaired_plan = candidate_plan
    else:
        repair_request = build_repair_request(grid, candidate_plan, verification)
        repair = repairer.repair(**repair_request)
        validate_repair_output(repair)
        repaired_plan = repair["repaired_plan"]

    final_result = executor.execute(grid=grid, plan=repaired_plan)
    return {
        "environment_id": grid["environment_id"],
        "generation": generation,
        "candidate_plan": candidate_plan,
        "verification": verification,
        "repair": repair,
        "repaired_plan": repaired_plan,
        "final_result": final_result,
    }
