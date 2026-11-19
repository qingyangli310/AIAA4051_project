from scripts.build_generation_like_repair_data import (
    generation_like_candidates,
    shortest_suffix,
    use_structured_feedback,
)
from src.executor.executor import execute_plan


GRID = {
    "environment_id": "repair-v3-grid",
    "rows": 3,
    "cols": 3,
    "start": [0, 0],
    "goal": [2, 2],
    "obstacles": [[1, 0]],
}


def test_localized_suffix_preserves_valid_prefix_and_reaches_goal():
    suffix = shortest_suffix(GRID, [0, 0])
    assert execute_plan(GRID, suffix)["success"] is True


def test_generation_like_candidates_are_train_only_collision_repairs():
    row = {
        **GRID,
        "base_sample_id": "base-repair-v3",
        "shortest_plan": ["right", "right", "down", "down"],
    }
    candidates = generation_like_candidates([row], "localized_suffix")
    assert candidates
    for candidate in candidates:
        assert candidate["executor_label"]["reason"] == "obstacle_collision"
        assert execute_plan(candidate["grid"], candidate["target_plan"])["success"]


def test_structured_feedback_prefix_does_not_leak_into_no_feedback():
    base = {
        "example_id": "repair-example",
        "feedback_source": "no_feedback",
        "verifier_feedback": None,
        "input_text": "Task: repair",
    }
    prompt = use_structured_feedback(base)["input_text"]
    assert "ERROR_STEP=unknown" in prompt
    assert "ERROR_REASON=unknown" in prompt
    assert "obstacle_collision" not in prompt
