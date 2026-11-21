import json
from collections import Counter
from pathlib import Path

from scripts.build_high_quality_repair_data import allocate_quotas
from src.executor.executor import execute_plan


ROOT = Path(__file__).resolve().parents[1]


def test_quality_quotas_balance_corruptions_and_reasons():
    quotas = allocate_quotas(500)
    corruption = Counter()
    reasons = Counter()
    for (corruption_type, reason), count in quotas.items():
        corruption[corruption_type] += count
        reasons[reason] += count
    assert corruption == {
        "premature_stopping": 125,
        "action_deletion": 125,
        "action_insertion": 125,
        "action_substitution": 125,
    }
    assert reasons == {
        "goal_not_reached": 375,
        "obstacle_collision": 90,
        "out_of_bounds": 35,
    }


def test_committed_quality_dataset_is_unique_and_executable():
    path = ROOT / "data/repair_high_quality_v1/repair_train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1000
    assert len({(row["input_text"], row["target_text"]) for row in rows}) == 1000
    assert Counter(row["feedback_source"] for row in rows) == {
        "no_feedback": 500,
        "executor_feedback": 500,
    }
    for row in rows:
        assert execute_plan(row["grid"], row["target_plan"])["success"]
        assert not execute_plan(row["grid"], row["candidate_plan"])["success"]
