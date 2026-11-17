"""Deterministic reference baselines for Part 1 algorithm validation."""

from __future__ import annotations

from collections import deque
from typing import Any

from src.executor.executor import ACTION_DELTAS, execute_plan


def shortest_path(grid: dict[str, Any]) -> list[str]:
    """Return one deterministic BFS shortest path, or raise if unreachable."""
    start, goal = tuple(grid["start"]), tuple(grid["goal"])
    obstacles = {tuple(cell) for cell in grid.get("obstacles", [])}
    order = ("up", "down", "left", "right")
    queue = deque([(start, [])])
    visited = {start}
    while queue:
        position, plan = queue.popleft()
        if position == goal:
            return plan
        for action in order:
            dr, dc = ACTION_DELTAS[action]
            nxt = (position[0] + dr, position[1] + dc)
            if not (0 <= nxt[0] < grid["rows"] and 0 <= nxt[1] < grid["cols"]):
                continue
            if nxt in obstacles or nxt in visited:
                continue
            visited.add(nxt)
            queue.append((nxt, [*plan, action]))
    raise ValueError(f"unreachable grid: {grid['environment_id']}")


class BFSGenerator:
    def generate(self, grid: dict[str, Any]) -> dict[str, Any]:
        plan = shortest_path(grid)
        return {
            "raw_output": ", ".join(plan),
            "plan": plan,
            "parse_success": True,
            "baseline": "symbolic_bfs_reference",
        }


class ExecutorVerifier:
    def verify(self, grid: dict[str, Any], plan: list[str]) -> dict[str, Any]:
        result = execute_plan(grid, plan)
        return {
            "valid": result["success"],
            "executor_valid": result["valid"],
            "first_error_step": result["first_error_step"],
            "reason": result["reason"],
            "baseline": "symbolic_executor_reference",
        }


class BFSRepairer:
    def repair(self, grid: dict[str, Any], **_: Any) -> dict[str, Any]:
        plan = shortest_path(grid)
        return {
            "raw_output": ", ".join(plan),
            "repaired_plan": plan,
            "parse_success": True,
            "baseline": "symbolic_bfs_reference",
        }
