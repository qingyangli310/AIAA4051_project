"""Load sanitized frozen candidates for paired pipeline experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from src.data.ppnl_io import load_records


CONDITIONS = {"id_single_error", "ood_double_error"}


def load_candidate_records(
    path: str | Path,
    *,
    condition: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load ID/OOD candidates without exposing reference plans to models."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(CONDITIONS)}")
    rows = load_records(path)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:limit]

    output: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} must be an object")
        grid = _normalize_grid(row)
        candidate = _candidate_plan(row)
        sample_id = row.get("sample_id", row.get("example_id", row.get("id")))
        base_sample_id = row.get("base_sample_id")
        environment_id = row.get("environment_id", grid["environment_id"])
        if not sample_id or not base_sample_id or not environment_id:
            raise ValueError(
                f"row {index} requires sample, base-sample, and environment IDs"
            )
        sample_id, base_sample_id, environment_id = (
            str(sample_id),
            str(base_sample_id),
            str(environment_id),
        )
        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)

        corruption = deepcopy(row.get("corruption_type"))
        if condition == "id_single_error":
            if isinstance(corruption, list):
                if len(corruption) != 1:
                    raise ValueError(f"ID row {sample_id} must describe one corruption")
            elif not isinstance(corruption, str) or not corruption:
                raise ValueError(f"ID row {sample_id} must identify one corruption")
        elif not isinstance(corruption, list) or len(corruption) != 2:
            raise ValueError(f"OOD row {sample_id} must describe two corruptions")

        output.append(
            {
                "sample_id": sample_id,
                "base_sample_id": base_sample_id,
                "environment_id": environment_id,
                "condition": condition,
                "grid": grid,
                "candidate_plan": candidate,
                "corruption_type": corruption,
                "optimal_length": _optimal_length(row),
            }
        )
    return output


def _candidate_plan(row: dict[str, Any]) -> list[str]:
    task_input = row.get("input")
    if not isinstance(task_input, dict):
        task_input = {}
    for container, key in (
        (task_input, "candidate_plan"),
        (row, "candidate_plan"),
        (task_input, "corrupted_plan"),
        (row, "corrupted_plan"),
    ):
        candidate = container.get(key)
        if candidate is not None:
            if not isinstance(candidate, list) or any(
                not isinstance(action, str) for action in candidate
            ):
                raise ValueError(f"{key} must be a list of action strings")
            return deepcopy(candidate)
    raise ValueError("record requires candidate_plan or corrupted_plan")


def _normalize_grid(row: dict[str, Any]) -> dict[str, Any]:
    task_input = row.get("input")
    if isinstance(task_input, dict) and isinstance(task_input.get("grid"), dict):
        source = task_input["grid"]
    else:
        source = row.get("grid", row)
    if not isinstance(source, dict):
        raise ValueError("record requires a grid object")
    grid = deepcopy(source)
    if "rows" not in grid or "cols" not in grid:
        size = grid.get("grid_size", grid.get("size"))
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError("grid requires rows/cols or a two-element grid_size")
        grid["rows"], grid["cols"] = size
    if not grid.get("environment_id"):
        grid["environment_id"] = row.get("environment_id")
    required = ("environment_id", "rows", "cols", "start", "goal", "obstacles")
    missing = [field for field in required if grid.get(field) is None]
    if missing:
        raise ValueError(f"grid is missing required fields: {missing}")
    return {field: deepcopy(grid[field]) for field in required}


def _optimal_length(row: dict[str, Any]) -> int | None:
    explicit = row.get("optimal_length")
    if isinstance(explicit, int) and explicit >= 0:
        return explicit
    for field in ("shortest_plan", "target_plan", "reference_actions", "original_plan"):
        plan = row.get(field)
        if isinstance(plan, list):
            return len(plan)
    return None
