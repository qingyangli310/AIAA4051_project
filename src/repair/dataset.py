"""Construct leak-free Repair examples from A's Train/Dev records."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy
from typing import Any

from src.repair_schema import FEEDBACK_SOURCES, build_repair_request


DEFAULT_FEEDBACK_SOURCES = ("no_feedback", "executor_feedback")
GRID_FIELDS = ("environment_id", "rows", "cols", "start", "goal", "obstacles")


def normalize_grid(record: dict[str, Any]) -> dict[str, Any]:
    """Return the frozen grid schema from either a task record or a raw grid."""
    candidate = record.get("input", {}).get("grid", record.get("grid", record))
    grid = deepcopy(candidate)
    if "rows" not in grid or "cols" not in grid:
        size = grid.get("grid_size", grid.get("size"))
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError("grid requires rows/cols or a two-element grid_size")
        grid["rows"], grid["cols"] = size

    normalized = {field: deepcopy(grid.get(field)) for field in GRID_FIELDS}
    missing = [field for field, value in normalized.items() if value is None]
    if missing:
        raise ValueError(f"grid is missing required fields: {missing}")
    return normalized


def build_repair_examples(
    records: Iterable[dict[str, Any]],
    *,
    feedback_sources: Sequence[str] = DEFAULT_FEEDBACK_SOURCES,
    default_split: str | None = None,
) -> list[dict[str, Any]]:
    """Build supervised Repair rows without placing the target in model input.

    ``no_feedback`` and ``executor_feedback`` are produced by default.  The
    ``self_feedback`` name is accepted so B's future verifier output can be
    plugged into the same schema, but it must already be present on the source
    row as ``self_feedback``.
    """
    unknown = set(feedback_sources) - FEEDBACK_SOURCES
    if unknown:
        raise ValueError(f"unsupported feedback sources: {sorted(unknown)}")

    examples: list[dict[str, Any]] = []
    for record in records:
        if record.get("task") not in (None, "Repair"):
            continue

        grid = normalize_grid(record)
        source_input = record.get("input", {})
        candidate_plan = source_input.get(
            "corrupted_plan",
            source_input.get(
                "candidate_plan",
                record.get("corrupted_plan", record.get("candidate_plan")),
            ),
        )
        if not isinstance(candidate_plan, list):
            raise ValueError("Repair source row requires a list-valued corrupted_plan")

        target = _repair_target(record)
        executor_feedback = deepcopy(
            source_input.get("verifier_feedback", record.get("executor_label"))
        )
        if not isinstance(executor_feedback, dict):
            raise ValueError("Repair source row requires executor/verifier feedback")
        executor_feedback = _normalize_invalid_feedback(executor_feedback)

        for feedback_source in feedback_sources:
            feedback = executor_feedback
            if feedback_source == "self_feedback":
                feedback = record.get("self_feedback")
                if not isinstance(feedback, dict):
                    raise ValueError(
                        "self_feedback was requested but is absent from the source row"
                    )
                feedback = _normalize_invalid_feedback(feedback)

            request = build_repair_request(
                grid,
                list(candidate_plan),
                feedback,
                feedback_source=feedback_source,
            )
            source_id = str(record.get("example_id", record.get("id", "unknown")))
            example = {
                "example_id": f"{source_id}__{feedback_source}",
                "task": "Repair",
                "split": record.get("split", default_split),
                "environment_id": grid["environment_id"],
                "base_sample_id": record.get("base_sample_id", source_id),
                "source_plan_id": record.get("source_plan_id", source_id),
                "feedback_source": feedback_source,
                "input": request,
                "output": {"repaired_plan": target},
                "provenance": {
                    "source_example_id": source_id,
                    "corruption_type": deepcopy(record.get("corruption_type")),
                    "corruption_seed": record.get("corruption_seed"),
                    "mutation_provenance": deepcopy(record.get("mutation_provenance")),
                },
            }
            _assert_no_target_leak(example)
            examples.append(example)
    return examples


def _repair_target(record: dict[str, Any]) -> list[str]:
    output = record.get("output", {})
    target = output.get(
        "repaired_plan",
        record.get(
            "repair_target", record.get("source_plan", record.get("original_plan"))
        ),
    )
    if not isinstance(target, list):
        raise ValueError("Repair source row requires a list-valued repair target")
    return list(target)


def _normalize_invalid_feedback(feedback: dict[str, Any]) -> dict[str, Any]:
    """Translate executor feasibility semantics into verifier task semantics."""
    result = {
        "valid": False,
        "first_error_step": feedback.get("first_error_step"),
        "reason": feedback.get("reason"),
    }
    if result["reason"] is None:
        raise ValueError("Repair feedback must describe an invalid candidate")
    return result


def _assert_no_target_leak(example: dict[str, Any]) -> None:
    request = example["input"]
    forbidden = {"repair_target", "source_plan", "optimal_plan", "repaired_plan"}
    leaked = forbidden.intersection(request)
    if leaked:
        raise ValueError(f"Repair input leaks target fields: {sorted(leaked)}")
