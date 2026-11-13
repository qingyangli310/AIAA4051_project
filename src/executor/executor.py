"""Canonical grid path executor used by Project 6 data and downstream modules."""
from __future__ import annotations

from typing import Any

ACTION_DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def execute_plan(grid: dict[str, Any], plan: str | list[str]) -> dict[str, Any]:
    """Execute a plan. Grid has grid_size, start, goal, obstacles (row/col order).

    ``valid`` means no movement/token error; ``success`` additionally requires goal.
    Action errors and goal_not_reached use 0-based indices. For goal_not_reached,
    the step is len(plan), the index immediately after the last action.
    """
    actions = plan.split() if isinstance(plan, str) else list(plan)
    actions = [a.strip().lower() if isinstance(a, str) else a for a in actions]
    rows, cols = grid.get("grid_size", grid.get("size", (grid.get("rows"), grid.get("cols"))))
    pos = tuple(grid["start"])
    goal = tuple(grid["goal"])
    obstacles = {tuple(p) for p in grid.get("obstacles", [])}
    trace = [list(pos)]
    result = {"valid": True, "success": False, "first_error_step": None, "reason": None,
              "final_position": list(pos), "executed_steps": 0, "trace": trace,
              "goal_not_reached_step_convention": "len(plan) (0-based)"}
    for step, action in enumerate(actions):
        if action not in ACTION_DELTAS:
            result.update(valid=False, first_error_step=step, reason="invalid_action_token")
            break
        dr, dc = ACTION_DELTAS[action]
        nxt = (pos[0] + dr, pos[1] + dc)
        if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
            result.update(valid=False, first_error_step=step, reason="out_of_bounds")
            break
        if nxt in obstacles:
            result.update(valid=False, first_error_step=step, reason="obstacle_collision")
            break
        pos = nxt
        result["executed_steps"] += 1
        trace.append(list(pos))
    result["final_position"] = list(pos)
    if result["valid"] and pos == goal:
        result["success"] = True
    elif result["valid"]:
        result.update(first_error_step=len(actions), reason="goal_not_reached")
    return result



ERROR_TYPES = {"obstacle_collision", "out_of_bounds", "invalid_action_token", "goal_not_reached"}


class Executor:
    """Adapter accepted by the existing C pipeline."""
    def execute(self, grid, plan):
        return execute_plan(grid, plan)
