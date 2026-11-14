"""Shared inference templates. Use these same templates for future training."""

PROTOCOL_VERSION = "member-b-day13-v3-bare-action-tokens"


def validate_grid(grid):
    """Check the input schema, not path feasibility (A owns the executor)."""
    if not isinstance(grid, dict):
        raise ValueError("grid must be a dictionary")
    required = {"environment_id", "rows", "cols", "start", "goal", "obstacles"}
    missing = required - grid.keys()
    if missing:
        raise ValueError(f"Missing grid fields: {sorted(missing)}")
    if not isinstance(grid["environment_id"], str) or not grid["environment_id"].strip():
        raise ValueError("environment_id must be a nonempty string")
    for field in ("rows", "cols"):
        if type(grid[field]) is not int or grid[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if not isinstance(grid["obstacles"], list):
        raise ValueError("obstacles must be a list")
    for cell in [grid["start"], grid["goal"], *grid["obstacles"]]:
        if not isinstance(cell, list) or len(cell) != 2 or any(type(x) is not int for x in cell):
            raise ValueError("coordinates must be [row, col] integer lists")
        if not (0 <= cell[0] < grid["rows"] and 0 <= cell[1] < grid["cols"]):
            raise ValueError("grid coordinates must be within bounds")
    if grid["start"] in grid["obstacles"] or grid["goal"] in grid["obstacles"]:
        raise ValueError("start and goal must not be obstacles")


def _grid_text(grid):
    validate_grid(grid)
    return (
        f"Grid size: {grid['rows']} rows and {grid['cols']} columns.\n"
        f"Start: {tuple(grid['start'])}.\n"
        f"Goal: {tuple(grid['goal'])}.\n"
        f"Obstacles: {[tuple(cell) for cell in grid['obstacles']]}.\n"
        "up decreases row by 1; down increases row by 1.\n"
        "left decreases col by 1; right increases col by 1.\n"
    )


def build_generation_prompt(grid):
    # Keeps the original 3x3 generation template text unchanged.
    return (
        "Find a path in the following grid.\n"
        "Coordinates are (row, col), starting from 0.\n"
        + _grid_text(grid)
        + "Avoid obstacles and stay inside the grid.\n"
        "Output only actions separated by commas: up, down, left, right.\n"
        "Path:"
    )


def build_verification_prompt(grid, plan):
    if not isinstance(plan, list) or any(not isinstance(action, str) for action in plan):
        raise ValueError("plan must be a list of strings; retain unknown action tokens")
    return (
        "Verify the candidate path in the following grid.\n"
        "Coordinates are (row, col), starting from 0 at the top left.\n"
        + _grid_text(grid)
        + "Only up, down, left, right are allowed actions.\n"
        "A valid path stays inside the grid, avoids obstacles, and ends at the goal.\n"
        "Action steps start at 0: the first action is step=0, the second is step=1.\n"
        "Report only the first failure.\n"
        f"If all {len(plan)} actions finish without reaching the goal, "
        f"use step={len(plan)} for goal_not_reached. "
        "This is the end checkpoint, not an extra action.\n"
        f"Candidate path: {', '.join(plan)}\n"
        "Output exactly valid or invalid | step=<integer> | reason=<error_type>.\n"
        "Allowed error types: obstacle_collision, out_of_bounds, "
        "invalid_action_token, goal_not_reached.\n"
        "Answer:"
    )
