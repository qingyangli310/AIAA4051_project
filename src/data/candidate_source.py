"""Load sanitized frozen candidates for paired pipeline experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from src.data.build_nov14_dataset import shortest_path
from src.data.environment import environment_id as canonical_environment_id
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

    output: list[dict[str, Any]] = []
    validate_candidate_ids(rows)
    for row in rows:
        grid = _normalize_grid(row)
        candidate = _candidate_plan(row)
        sample_id = _sample_id(row)
        base_sample_id = row["base_sample_id"]
        environment_id = grid["environment_id"]

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
    return output[:limit]


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
    rows, cols = grid["rows"], grid["cols"]
    if any(type(n) is not int or n <= 0 for n in (rows, cols)):
        raise ValueError("grid dimensions must be positive integers")
    for alias in ("grid_size", "size"):
        if alias in grid and grid[alias] not in ([rows, cols], (rows, cols)):
            raise ValueError("grid dimensions disagree with size alias")
    if not isinstance(grid["obstacles"], (list, tuple)):
        raise ValueError("obstacles must be a coordinate sequence")
    for point in (grid["start"], grid["goal"], *grid["obstacles"]):
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or any(type(n) is not int for n in point)
                or not (0 <= point[0] < rows and 0 <= point[1] < cols)):
            raise ValueError("invalid grid coordinate")
    obstacles = sorted([list(point) for point in grid["obstacles"]])
    if len(set(map(tuple, obstacles))) != len(obstacles):
        raise ValueError("duplicate obstacles")
    for key in ("start", "goal"):
        grid[key] = list(grid[key])
        if grid[key] in obstacles:
            raise ValueError(f"{key} is blocked")
    grid["obstacles"] = obstacles
    eid = grid["environment_id"]
    if row.get("environment_id", eid) != eid:
        raise ValueError("row and grid environment_id disagree")
    if eid != canonical_environment_id(rows, cols, obstacles):
        raise ValueError("environment_id does not match canonical grid")
    return {field: grid[field] for field in required}


def _sample_id(row):
    return row.get("sample_id", row.get("example_id", row.get("id")))


def validate_candidate_ids(rows):
    """Check identity across the entire condition, including rows beyond a limit."""
    seen = {"sample_id": set(), "base_sample_id": set()}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("candidate row must be an object")
        for key, value in (("sample_id", _sample_id(row)),
                           ("base_sample_id", row.get("base_sample_id"))):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"invalid {key}")
            if value in seen[key]:
                raise ValueError(f"duplicate {key}: {value}")
            seen[key].add(value)


def validate_candidate_grid(record, runtime_grid):
    validate_candidate_ids([record])
    if _normalize_grid(record) != _normalize_grid({"grid": runtime_grid}):
        raise ValueError("candidate grid differs from runtime grid")


def _optimal_length(row: dict[str, Any]) -> int | None:
    explicit = [row[key] for key in ("shortest_length", "optimal_length")
                if row.get(key) is not None]
    if not explicit:
        return None
    shortest = shortest_path(_normalize_grid(row))
    if shortest is None or any(type(n) is not int or n != len(shortest) for n in explicit):
        raise ValueError("declared optimal length disagrees with BFS")
    return explicit[0]
