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
        "grid": grid,
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


def executor_feedback(result: dict[str, Any]) -> dict[str, Any]:
    """Convert canonical Executor output to the Verification task contract."""
    return {
        "valid": result.get("success") is True,
        "executor_valid": result.get("valid"),
        "first_error_step": result.get("first_error_step"),
        "reason": result.get("reason"),
        "parse_success": True,
        "feedback_kind": "executor_ground_truth",
    }


def _run_repair_branch(
    *,
    grid: dict[str, Any],
    candidate_plan: list[str],
    feedback_source: str,
    feedback: dict[str, Any],
    repairer: Any,
    executor: Any,
) -> dict[str, Any]:
    """Run one repair condition while preserving parse and routing failures."""
    if feedback.get("valid") is True:
        final_result = executor.execute(grid=grid, plan=candidate_plan)
        return {
            "feedback_source": feedback_source,
            "verification": feedback,
            "repair": None,
            "repaired_plan": candidate_plan,
            "final_result": final_result,
            "routing_status": "candidate_accepted",
        }

    if feedback.get("parse_success") is False or feedback.get("valid") is not False:
        final_result = executor.execute(grid=grid, plan=candidate_plan)
        return {
            "feedback_source": feedback_source,
            "verification": feedback,
            "repair": None,
            "repaired_plan": candidate_plan,
            "final_result": final_result,
            "routing_status": "verifier_parse_failed",
        }

    repair_request = build_repair_request(
        grid,
        candidate_plan,
        feedback,
        feedback_source=feedback_source,
    )
    repair = repairer.repair(**repair_request)
    validate_repair_output(repair)
    repaired_plan = repair["repaired_plan"] if repair["parse_success"] else []
    final_result = executor.execute(grid=grid, plan=repaired_plan)
    return {
        "feedback_source": feedback_source,
        "verification": feedback,
        "repair": repair,
        "repaired_plan": repaired_plan,
        "final_result": final_result,
        "routing_status": "repair_attempted",
    }


def run_pipeline_comparison(
    grid: dict[str, Any],
    generator: Any,
    verifier: Any,
    repairer: Any,
    executor: Any,
    *,
    conditions: tuple[str, ...] = (
        "direct",
        "self_feedback",
        "executor_feedback",
    ),
) -> dict[str, Any]:
    """Generate once and compare feedback conditions on the same candidate.

    ``direct`` never calls Repair. ``self_feedback`` routes using the learned
    Verifier. ``executor_feedback`` routes and diagnoses using canonical
    Executor ground truth. ``no_feedback`` uses Executor only to select failed
    candidates and hides the diagnosis from Repair.
    """
    supported = {"direct", "no_feedback", "self_feedback", "executor_feedback"}
    unknown = set(conditions) - supported
    if unknown:
        raise ValueError(f"unsupported comparison conditions: {sorted(unknown)}")
    if len(set(conditions)) != len(conditions):
        raise ValueError("comparison conditions must be unique")

    generation = generator.generate(grid)
    candidate_plan = generation.get("plan")
    if not isinstance(candidate_plan, list):
        raise ValueError("generator output must contain plan as a list")
    candidate_result = executor.execute(grid=grid, plan=candidate_plan)
    ground_truth_feedback = executor_feedback(candidate_result)

    branches: dict[str, Any] = {}
    if "direct" in conditions:
        branches["direct"] = {
            "feedback_source": "direct",
            "verification": None,
            "repair": None,
            "repaired_plan": candidate_plan,
            "final_result": candidate_result,
            "routing_status": "direct_generation",
        }
    if "self_feedback" in conditions:
        learned_feedback = verifier.verify(grid=grid, plan=candidate_plan)
        branches["self_feedback"] = _run_repair_branch(
            grid=grid,
            candidate_plan=candidate_plan,
            feedback_source="self_feedback",
            feedback=learned_feedback,
            repairer=repairer,
            executor=executor,
        )
    if "executor_feedback" in conditions:
        branches["executor_feedback"] = _run_repair_branch(
            grid=grid,
            candidate_plan=candidate_plan,
            feedback_source="executor_feedback",
            feedback=ground_truth_feedback,
            repairer=repairer,
            executor=executor,
        )
    if "no_feedback" in conditions:
        branches["no_feedback"] = _run_repair_branch(
            grid=grid,
            candidate_plan=candidate_plan,
            feedback_source="no_feedback",
            feedback=ground_truth_feedback,
            repairer=repairer,
            executor=executor,
        )

    return {
        "environment_id": grid["environment_id"],
        "grid": grid,
        "generation": generation,
        "candidate_plan": candidate_plan,
        "candidate_executor": candidate_result,
        "branches": branches,
    }


def run_pipeline_comparison_batch(
    rows: list[dict[str, Any]],
    generator: Any,
    verifier: Any,
    repairer: Any,
    executor: Any,
    *,
    conditions: tuple[str, ...] = (
        "direct",
        "self_feedback",
        "executor_feedback",
    ),
) -> list[dict[str, Any]]:
    """Run paired comparisons and isolate individual sample failures."""
    include_shuffled = "shuffled_executor_feedback" in conditions
    base_conditions = tuple(
        condition for condition in conditions if condition != "shuffled_executor_feedback"
    )
    records: list[dict[str, Any]] = []
    for row in rows:
        grid = row.get("grid", row)
        try:
            result = run_pipeline_comparison(
                grid,
                generator,
                verifier,
                repairer,
                executor,
                conditions=base_conditions,
            )
            records.append(
                {
                    "status": "ok",
                    "base_sample_id": row.get("base_sample_id"),
                    "example_id": row.get("example_id", row.get("sample_id")),
                    "sample_id": row.get("sample_id"),
                    "corruption_type": row.get("corruption_type"),
                    "test_condition": row.get("condition"),
                    "optimal_length": _row_optimal_length(row),
                    **result,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "status": "error",
                    "base_sample_id": row.get("base_sample_id"),
                    "example_id": row.get("example_id", row.get("sample_id")),
                    "environment_id": grid.get("environment_id"),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    if include_shuffled:
        completed = [record for record in records if record["status"] == "ok"]
        failed_by_length: dict[int, list[dict[str, Any]]] = {}
        for record in completed:
            if record["candidate_executor"].get("success") is not True:
                failed_by_length.setdefault(len(record["candidate_plan"]), []).append(record)

        feedback_source_by_record: dict[int, dict[str, Any]] = {}
        for group in failed_by_length.values():
            for index, target in enumerate(group):
                own_key = (
                    target["candidate_executor"].get("first_error_step"),
                    target["candidate_executor"].get("reason"),
                )
                candidates = group[index + 1 :] + group[: index + 1]
                source = next(
                    (
                        candidate
                        for candidate in candidates
                        if (
                            candidate["candidate_executor"].get("first_error_step"),
                            candidate["candidate_executor"].get("reason"),
                        )
                        != own_key
                    ),
                    candidates[0],
                )
                feedback_source_by_record[id(target)] = source

        for record in completed:
            own_feedback = executor_feedback(record["candidate_executor"])
            source = feedback_source_by_record.get(id(record), record)
            shuffled_feedback = executor_feedback(source["candidate_executor"])
            branch = _run_repair_branch(
                grid=record["grid"],
                candidate_plan=record["candidate_plan"],
                feedback_source="executor_feedback",
                feedback=(
                    shuffled_feedback
                    if record["candidate_executor"].get("success") is not True
                    else own_feedback
                ),
                repairer=repairer,
                executor=executor,
            )
            branch["condition"] = "shuffled_executor_feedback"
            branch["feedback_source_environment_id"] = source["environment_id"]
            branch["feedback_shuffled"] = source is not record
            branch["feedback_content_changed"] = (
                shuffled_feedback.get("first_error_step"), shuffled_feedback.get("reason")
            ) != (own_feedback.get("first_error_step"), own_feedback.get("reason"))
            record["branches"]["shuffled_executor_feedback"] = branch
    return records


def _row_optimal_length(row: dict[str, Any]) -> int | None:
    """Read the optimal path length from every frozen task representation."""
    explicit = row.get("optimal_length")
    if isinstance(explicit, int) and explicit >= 0:
        return explicit
    for field in ("shortest_plan", "target_plan", "reference_actions"):
        plan = row.get(field)
        if isinstance(plan, list):
            return len(plan)
    return None
