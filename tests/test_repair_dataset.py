import copy

import pytest

from src.repair.dataset import build_repair_examples


SOURCE = {
    "example_id": "train_env1_repair",
    "split": "train",
    "environment_id": "env1",
    "base_sample_id": "base1",
    "corruption_type": ["premature_stopping"],
    "task": "Repair",
    "input": {
        "grid": {
            "environment_id": "env1",
            "grid_size": [3, 3],
            "start": [0, 0],
            "goal": [0, 2],
            "obstacles": [],
        },
        "corrupted_plan": ["right"],
        # A's executor uses valid for movement feasibility; Repair task validity
        # is normalized from success/reason instead.
        "verifier_feedback": {
            "valid": True,
            "success": False,
            "first_error_step": 1,
            "reason": "goal_not_reached",
        },
    },
    "output": {"repaired_plan": ["right", "right"]},
}


def test_builds_two_default_feedback_conditions_without_target_leak() -> None:
    source = copy.deepcopy(SOURCE)
    examples = build_repair_examples([source])

    assert [example["feedback_source"] for example in examples] == [
        "no_feedback",
        "executor_feedback",
    ]
    assert examples[0]["input"]["verifier_feedback"] is None
    assert examples[1]["input"]["verifier_feedback"] == {
        "valid": False,
        "first_error_step": 1,
        "reason": "goal_not_reached",
    }
    assert examples[0]["output"]["repaired_plan"] == ["right", "right"]
    assert "repaired_plan" not in examples[0]["input"]
    assert "source_plan" not in examples[0]["input"]
    assert source == SOURCE


def test_skips_non_repair_task_rows() -> None:
    generation = copy.deepcopy(SOURCE)
    generation["task"] = "Generation"
    assert build_repair_examples([generation]) == []


def test_self_feedback_is_reserved_but_requires_real_content() -> None:
    with pytest.raises(ValueError, match="self_feedback"):
        build_repair_examples([SOURCE], feedback_sources=["self_feedback"])


def test_self_feedback_can_be_added_without_schema_change() -> None:
    source = copy.deepcopy(SOURCE)
    source["self_feedback"] = {
        "valid": False,
        "first_error_step": 1,
        "reason": "goal_not_reached",
    }
    example = build_repair_examples(
        [source], feedback_sources=["self_feedback"]
    )[0]
    assert example["input"]["verifier_feedback"]["reason"] == "goal_not_reached"


def test_rejects_missing_repair_target() -> None:
    source = copy.deepcopy(SOURCE)
    source["output"] = {}
    with pytest.raises(ValueError, match="repair target"):
        build_repair_examples([source])


def test_accepts_member_a_nov14_single_error_schema() -> None:
    source = {
        "environment_id": "env_a",
        "base_sample_id": "base_a",
        "source_plan_id": "base_a:shortest",
        "grid": {
            "environment_id": "env_a",
            "rows": 6,
            "cols": 6,
            "start": [1, 4],
            "goal": [2, 1],
            "obstacles": [[5, 3]],
        },
        "corruption_type": "action_deletion",
        "corruption_position": 3,
        "corruption_seed": 1195427521,
        "original_plan": ["down", "left", "left", "left"],
        "corrupted_plan": ["down", "left", "left"],
        "executor_label": {
            "valid": True,
            "success": False,
            "first_error_step": 3,
            "reason": "goal_not_reached",
        },
    }

    examples = build_repair_examples([source], default_split="train")
    assert len(examples) == 2
    assert examples[0]["split"] == "train"
    assert examples[0]["base_sample_id"] == "base_a"
    assert examples[0]["source_plan_id"] == "base_a:shortest"
    assert examples[0]["input"]["candidate_plan"] == ["down", "left", "left"]
    assert examples[0]["output"]["repaired_plan"] == [
        "down",
        "left",
        "left",
        "left",
    ]
    assert examples[0]["provenance"]["corruption_seed"] == 1195427521
