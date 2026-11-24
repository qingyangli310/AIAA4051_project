"""End-to-end Generate-Verify-Repair-Execute orchestration."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
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
    candidate_source: str = "generator",
    candidate_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one example from either a fresh Generator or a frozen dataset candidate.

    Dataset records are read-only inputs.  The complete source row is never
    passed to model components: only the grid and candidate plan are exposed.
    """
    if candidate_source not in {"generator", "dataset"}:
        raise ValueError("candidate_source must be 'generator' or 'dataset'")

    sample_metadata = _sample_metadata(candidate_record, grid)
    if candidate_source == "generator":
        if candidate_record is not None:
            raise ValueError("candidate_record is only valid with candidate_source='dataset'")
        if generator is None:
            raise ValueError("generator is required when candidate_source='generator'")
        generation = generator.generate(deepcopy(grid))
        candidate_plan = generation.get("plan")
        if not isinstance(candidate_plan, list):
            raise ValueError("generator output must contain plan as a list")
        candidate_plan = deepcopy(candidate_plan)
    else:
        if not isinstance(candidate_record, dict):
            raise ValueError("candidate_record is required when candidate_source='dataset'")
        generation = None
        candidate_plan = _read_dataset_candidate(candidate_record)

    verification = verifier.verify(grid=deepcopy(grid), plan=deepcopy(candidate_plan))

    repair = None
    if verification["valid"]:
        repaired_plan = candidate_plan
    else:
        repair_request = build_repair_request(
            deepcopy(grid),
            deepcopy(candidate_plan),
            deepcopy(verification),
            feedback_source=feedback_source,
        )
        repair = repairer.repair(**repair_request)
        validate_repair_output(repair)
        repaired_plan = repair["repaired_plan"]

    final_result = executor.execute(grid=deepcopy(grid), plan=deepcopy(repaired_plan))
    return {
        **sample_metadata,
        "environment_id": sample_metadata["environment_id"],
        "feedback_source": feedback_source,
        "candidate_source": candidate_source,
        "generation": generation,
        "candidate_plan": deepcopy(candidate_plan),
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
    candidate_source: str = "generator",
    candidate_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run a batch while preserving per-sample failures instead of aborting."""
    if candidate_source == "dataset":
        if candidate_records is None or len(candidate_records) != len(grids):
            raise ValueError("dataset mode requires one candidate record per grid")
    elif candidate_records is not None:
        raise ValueError("candidate_records may only be supplied in dataset mode")

    records: list[dict[str, Any]] = []
    for index, grid in enumerate(grids):
        candidate_record = candidate_records[index] if candidate_records is not None else None
        try:
            result = run_pipeline(
                grid,
                generator,
                verifier,
                repairer,
                executor,
                feedback_source=feedback_source,
                candidate_source=candidate_source,
                candidate_record=candidate_record,
            )
            records.append({"status": "ok", **result})
        except Exception as exc:  # The caller receives an auditable failure record.
            records.append(
                {
                    "status": "error",
                    **_sample_metadata(candidate_record, grid),
                    "environment_id": grid.get("environment_id"),
                    "feedback_source": feedback_source,
                    "candidate_source": candidate_source,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
    return records


def run_pipeline_comparison(
    grid: dict[str, Any],
    generator: Any,
    verifier: Any,
    repairer: Any,
    executor: Any,
    *,
    conditions: tuple[str, ...] = ("direct", "self_feedback", "executor_feedback"),
    candidate_source: str = "generator",
    candidate_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare Direct, Self-feedback, and Executor-feedback on one candidate.

    In dataset mode the candidate is read exactly once and Generator is never
    invoked.  Every condition receives a defensive copy of that same plan.
    """
    supported = {"direct", "self_feedback", "executor_feedback"}
    unknown = set(conditions) - supported
    if unknown:
        raise ValueError(f"unsupported comparison conditions: {sorted(unknown)}")
    if len(set(conditions)) != len(conditions):
        raise ValueError("comparison conditions must be unique")

    if candidate_source not in {"generator", "dataset"}:
        raise ValueError("candidate_source must be 'generator' or 'dataset'")
    if candidate_source == "generator":
        if generator is None:
            raise ValueError("generator is required when candidate_source='generator'")
        if candidate_record is not None:
            raise ValueError("candidate_record is only valid with candidate_source='dataset'")
        generation = generator.generate(deepcopy(grid))
        candidate_plan = generation.get("plan")
        if not isinstance(candidate_plan, list):
            raise ValueError("generator output must contain plan as a list")
        candidate_plan = deepcopy(candidate_plan)
    else:
        if not isinstance(candidate_record, dict):
            raise ValueError("candidate_record is required when candidate_source='dataset'")
        generation = None
        candidate_plan = _read_dataset_candidate(candidate_record)

    candidate_hash = hashlib.sha256(
        json.dumps(candidate_plan, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    branches: dict[str, Any] = {}

    if "direct" in conditions:
        direct_result = executor.execute(grid=deepcopy(grid), plan=deepcopy(candidate_plan))
        branches["direct"] = {
            "feedback_source": "direct",
            "verification": None,
            "repair": None,
            "repaired_plan": deepcopy(candidate_plan),
            "final_result": direct_result,
            "routing_status": "direct_candidate_execution",
        }

    if "self_feedback" in conditions:
        learned_feedback = verifier.verify(
            grid=deepcopy(grid), plan=deepcopy(candidate_plan)
        )
        branches["self_feedback"] = _run_feedback_repair(
            grid, candidate_plan, learned_feedback, "self_feedback", repairer, executor
        )

    if "executor_feedback" in conditions:
        initial_result = executor.execute(
            grid=deepcopy(grid), plan=deepcopy(candidate_plan)
        )
        executor_diagnosis = {
            "valid": initial_result.get("success") is True,
            "executor_valid": initial_result.get("valid"),
            "first_error_step": initial_result.get("first_error_step"),
            "reason": initial_result.get("reason"),
        }
        branches["executor_feedback"] = _run_feedback_repair(
            grid, candidate_plan, executor_diagnosis, "executor_feedback", repairer, executor
        )

    return {
        **_sample_metadata(candidate_record, grid),
        "candidate_source": candidate_source,
        "generation": generation,
        "candidate_plan": deepcopy(candidate_plan),
        "candidate_plan_sha256": candidate_hash,
        "branches": branches,
    }


def run_pipeline_comparison_batch(
    records: list[dict[str, Any]],
    generator: Any,
    verifier: Any,
    repairer: Any,
    executor: Any,
    *,
    conditions: tuple[str, ...] = ("direct", "self_feedback", "executor_feedback"),
    candidate_source: str = "dataset",
) -> list[dict[str, Any]]:
    """Run paired conditions over records while preserving per-row IDs/errors."""
    results: list[dict[str, Any]] = []
    for row in records:
        try:
            grid = _record_grid(row)
            result = run_pipeline_comparison(
                grid,
                generator,
                verifier,
                repairer,
                executor,
                conditions=conditions,
                candidate_source=candidate_source,
                candidate_record=row if candidate_source == "dataset" else None,
            )
            results.append({"status": "ok", **result})
        except Exception as exc:
            results.append(
                {
                    "status": "error",
                    **_sample_metadata(row, row.get("grid", row)),
                    "candidate_source": candidate_source,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
    return results


def _run_feedback_repair(
    grid: dict[str, Any],
    candidate_plan: list[str],
    feedback: dict[str, Any],
    feedback_source: str,
    repairer: Any,
    executor: Any,
) -> dict[str, Any]:
    if feedback.get("valid") is True:
        final_result = executor.execute(grid=deepcopy(grid), plan=deepcopy(candidate_plan))
        return {
            "feedback_source": feedback_source,
            "verification": deepcopy(feedback),
            "repair": None,
            "repaired_plan": deepcopy(candidate_plan),
            "final_result": final_result,
            "routing_status": "candidate_accepted",
        }
    if feedback.get("valid") is not False:
        final_result = executor.execute(grid=deepcopy(grid), plan=deepcopy(candidate_plan))
        return {
            "feedback_source": feedback_source,
            "verification": deepcopy(feedback),
            "repair": None,
            "repaired_plan": deepcopy(candidate_plan),
            "final_result": final_result,
            "routing_status": "feedback_unusable",
        }
    request = build_repair_request(
        deepcopy(grid), deepcopy(candidate_plan), deepcopy(feedback), feedback_source
    )
    repair = repairer.repair(**request)
    validate_repair_output(repair)
    repaired_plan = deepcopy(repair["repaired_plan"]) if repair["parse_success"] else []
    final_result = executor.execute(grid=deepcopy(grid), plan=deepcopy(repaired_plan))
    return {
        "feedback_source": feedback_source,
        "verification": deepcopy(feedback),
        "repair": repair,
        "repaired_plan": repaired_plan,
        "final_result": final_result,
        "routing_status": "repair_attempted",
    }


def _read_dataset_candidate(record: dict[str, Any]) -> list[str]:
    task_input = record.get("input")
    if not isinstance(task_input, dict):
        task_input = {}
    for container, key in (
        (task_input, "candidate_plan"),
        (record, "candidate_plan"),
        (task_input, "corrupted_plan"),
        (record, "corrupted_plan"),
    ):
        value = container.get(key)
        if value is not None:
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise ValueError(f"dataset {key} must be a list of action strings")
            return deepcopy(value)
    raise ValueError("dataset record requires candidate_plan or corrupted_plan")


def _record_grid(record: dict[str, Any]) -> dict[str, Any]:
    task_input = record.get("input")
    if isinstance(task_input, dict) and isinstance(task_input.get("grid"), dict):
        grid = task_input["grid"]
    else:
        grid = record.get("grid", record)
    if not isinstance(grid, dict):
        raise ValueError("dataset row must contain a grid object")
    normalized = deepcopy(grid)
    if "rows" not in normalized or "cols" not in normalized:
        size = normalized.get("grid_size", normalized.get("size"))
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError("grid requires rows/cols or a two-element grid_size")
        normalized["rows"], normalized["cols"] = size
    if not normalized.get("environment_id"):
        normalized["environment_id"] = record.get("environment_id")
    return normalized


def _sample_metadata(
    record: dict[str, Any] | None, grid: dict[str, Any]
) -> dict[str, Any]:
    record = record or {}
    environment_id = record.get("environment_id")
    if environment_id is None and isinstance(grid, dict):
        environment_id = grid.get("environment_id")
    return {
        "sample_id": record.get("sample_id", record.get("example_id")),
        "base_sample_id": record.get("base_sample_id"),
        "environment_id": environment_id,
    }
