import json
from pathlib import Path

from src.models.output_parser import parse_plan, parse_trace, parse_verification
from src.models.prompts import build_generation_prompt, build_repair_prompt
from src.part1.baselines import shortest_path
from src.part1.data import build_part1_datasets
from src.part1.evaluate import score_predictions
from scripts.train_part1 import diagnostic_rows, sha256_file


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


def test_trace_parser_and_scoring_execute_actions():
    raw = "Trace: (0,0) down (1,0) down (2,0) right (2,1) right (2,2)"
    parsed = parse_trace(raw, GRID)
    assert parsed["parse_success"] is True
    assert parsed["plan"] == ["down", "down", "right", "right"]
    inconsistent = "Trace: (0,0) down (2,0)"
    assert parse_trace(inconsistent, GRID)["parse_success"] is False
    row = {
        "grid": GRID,
        "target_text": raw,
        "target_plan": ["down", "down", "right", "right"],
        "output_format": "trace_only",
    }
    report = score_predictions("repair", [row], [raw])
    assert report["parse_rate"] == 1.0
    assert report["metrics"]["repair_success_rate"] == 1.0


def test_no_feedback_prompt_hides_executor_diagnosis():
    prompt = build_repair_prompt(GRID, ["down"], "no_feedback", None)
    assert "Verifier feedback: none" in prompt
    assert "goal_not_reached" not in prompt


def test_generation_prompt_contains_spatial_map_and_relative_goal():
    prompt = build_generation_prompt(GRID)
    assert "delta_row=+2, delta_col=+2" in prompt
    assert "S . .\n. # .\n. . G" in prompt


def test_overfit_probe_is_first_action_balanced():
    rows = [json.loads(line) for line in (ROOT / "data/part1/generation_train.jsonl").read_text().splitlines()]
    selected = diagnostic_rows(rows, 4)
    assert [row["target_plan"] for row in selected] == [["up"], ["down"], ["left"], ["right"]]


def test_sha256_file_streams_expected_digest(tmp_path):
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"verified local model snapshot")
    assert sha256_file(artifact) == "579f3bfcad27312250dbdd674bdc6dfd23d5af2d820682c4389ea5e141238ec0"


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
