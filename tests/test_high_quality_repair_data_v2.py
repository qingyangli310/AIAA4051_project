import json
from collections import Counter
from pathlib import Path

from src.executor.executor import execute_plan


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/repair_high_quality_v2"


def _rows(name: str):
    return [json.loads(line) for line in (DATA / name).read_text().splitlines()]


def test_v2_train_has_500_independent_paired_candidates():
    rows = _rows("repair_train.jsonl")
    assert len(rows) == 1000
    assert len({row["base_sample_id"] for row in rows}) == 500
    assert len({row["environment_id"] for row in rows}) == 500
    assert Counter(row["feedback_source"] for row in rows) == {
        "no_feedback": 500,
        "executor_feedback": 500,
    }
    assert Counter(row["corruption_type"] for row in rows) == {
        "action_substitution": 250,
        "action_deletion": 250,
        "action_insertion": 250,
        "premature_stopping": 250,
    }
    by_base = Counter(row["base_sample_id"] for row in rows)
    assert set(by_base.values()) == {2}
    assert len({(row["input_text"], row["target_text"]) for row in rows}) == 1000


def test_v2_dev_has_50_independent_examples():
    rows = _rows("repair_dev.jsonl")
    assert len(rows) == 50
    assert len({row["base_sample_id"] for row in rows}) == 50
    assert len({row["environment_id"] for row in rows}) == 50
    assert Counter(row["feedback_source"] for row in rows) == {
        "no_feedback": 25,
        "executor_feedback": 25,
    }


def test_v2_is_cross_task_disjoint_and_executor_validated():
    train = _rows("repair_train.jsonl")
    dev = _rows("repair_dev.jsonl")
    generation_train = _rows(
        "../../artifacts/generation_trace_v9/data/transformed_train.jsonl"
    )
    generation_dev = _rows(
        "../../artifacts/generation_trace_v9/data/transformed_dev.jsonl"
    )
    verification_train = _rows(
        "../verification_high_quality_v1/verification_train.jsonl"
    )
    verification_dev = _rows(
        "../verification_high_quality_v1/verification_dev.jsonl"
    )

    assert not ({r["base_sample_id"] for r in train} & {
        r["base_sample_id"] for r in generation_train + verification_train
    })
    assert not ({r["base_sample_id"] for r in dev} & {
        r["base_sample_id"] for r in generation_dev + verification_dev
    })
    assert not ({r["environment_id"] for r in train} & {
        r["environment_id"] for r in dev
    })
    for row in train + dev:
        assert not execute_plan(row["grid"], row["candidate_plan"])["success"]
        assert execute_plan(row["grid"], row["target_plan"])["success"]
