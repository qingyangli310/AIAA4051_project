import json
from pathlib import Path

from src.models.output_parser import parse_plan, parse_verification
from src.models.prompts import build_generation_prompt, build_repair_prompt
from src.part1.baselines import shortest_path
from src.part1.data import build_part1_datasets
from src.part1.evaluate import score_predictions


ROOT = Path(__file__).resolve().parents[1]
GRID = {
    "environment_id": "part1-grid",
    "rows": 3,
    "cols": 3,
    "start": [0, 0],
    "goal": [2, 2],
    "obstacles": [[1, 1]],
}


def test_shortest_path_is_valid_and_deterministic():
    assert shortest_path(GRID) == ["down", "down", "right", "right"]


def test_strict_plan_and_verification_parsers():
    assert parse_plan("down, right")["plan"] == ["down", "right"]
    assert parse_plan("[down, right]")["parse_success"] is False
    valid = parse_verification("valid", 2)
    assert valid["parse_success"] and valid["valid"]
    invalid = parse_verification("invalid | step=2 | reason=goal_not_reached", 2)
    assert invalid["parse_success"] and invalid["valid"] is False
    assert parse_verification("invalid | step=1 | reason=goal_not_reached", 2)["parse_success"] is False


def test_no_feedback_prompt_hides_executor_diagnosis():
    prompt = build_repair_prompt(GRID, ["down"], "no_feedback", None)
    assert "Verifier feedback: none" in prompt
    assert "goal_not_reached" not in prompt


def test_generation_prompt_contains_spatial_map_and_relative_goal():
    prompt = build_generation_prompt(GRID)
    assert "delta_row=+2, delta_col=+2" in prompt
    assert "S . .\n. # .\n. . G" in prompt


def test_part1_data_is_reproducible_and_respects_budgets(tmp_path):
    first = build_part1_datasets(ROOT, tmp_path / "first", 42)
    second = build_part1_datasets(ROOT, tmp_path / "second", 42)
    assert first == second
    assert first["train_total"] == 4000
    assert first["dev_total"] == 200
    assert first["counts"]["verification_id_test"] == 200
    assert all(value == 0 for value in first["environment_intersections"].values())
    assert first["generation_train_first_action_counts"] == {
        "up": 250,
        "down": 250,
        "left": 250,
        "right": 250,
    }


def test_generation_scoring_detects_output_collapse():
    rows = [
        {
            "grid": GRID,
            "target_text": "down, down, right, right",
            "target_plan": ["down", "down", "right", "right"],
        },
        {
            "grid": {**GRID, "environment_id": "other", "goal": [0, 2]},
            "target_text": "right, right",
            "target_plan": ["right", "right"],
        },
    ]
    report = score_predictions("generation", rows, ["up, up", "up, up"])
    assert report["unique_predictions"] == 1
    assert report["most_common_share"] == 1.0
    assert report["metrics"]["success_rate"] == 0.0


def test_generated_manifest_matches_committed_data():
    manifest = json.loads((ROOT / "data/part1/manifest.json").read_text())
    assert manifest["train_total"] == 4000
    assert manifest["dev_total"] == 200
