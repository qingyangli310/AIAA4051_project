from scripts.build_generation_like_repair_data import (
    deduplicate_candidate_model_io,
    generation_like_candidates,
    shortest_suffix,
    use_structured_feedback,
)
from scripts.build_feedback_localized_repair_data import controlled_candidates
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


def test_candidate_deduplication_ignores_distinct_source_metadata():
    candidate = {
        "base_sample_id": "base-one",
        "environment_id": GRID["environment_id"],
        "grid": GRID,
        "candidate_plan": ["down", "down"],
        "target_plan": ["right", "right", "down", "down"],
    }
    duplicate = {**candidate, "base_sample_id": "base-two"}
    distinct = {**candidate, "candidate_plan": ["right", "down"]}
    result = deduplicate_candidate_model_io([candidate, duplicate, distinct])
    assert result == [candidate, distinct]


def test_controlled_candidate_target_preserves_prefix_and_repairs_suffix():
    row = {
        "base_sample_id": "controlled-base",
        "environment_id": GRID["environment_id"],
        "grid": GRID,
        "candidate_plan": ["right", "down", "down"],
        "corruption_type": "action_substitution",
    }
    [candidate] = controlled_candidates([row])
    error_step = candidate["executor_label"]["first_error_step"]
    assert candidate["target_plan"][:error_step] == row["candidate_plan"][:error_step]
    assert execute_plan(GRID, candidate["target_plan"])["success"] is True
