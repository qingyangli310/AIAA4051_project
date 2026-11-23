import json
from pathlib import Path

from src.executor.executor import execute_plan


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/verification_high_quality_v1"


def _read(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _semantic(rows):
    return {
        (row["environment_id"], tuple(plan))
        for row in rows
        for plan in (
            row.get("reference_actions"),
            row.get("target_plan"),
            row.get("candidate_plan"),
        )
        if isinstance(plan, list)
    }


def test_high_quality_verification_manifest_and_balances():
    train = _read(DATA / "verification_train.jsonl")
    dev = _read(DATA / "verification_dev.jsonl")
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))

    assert len(train) == manifest["train"]["count"] == 1000
    assert len(dev) == manifest["dev"]["count"] == 50
    assert manifest["train"]["label_counts"] == {"valid": 500, "invalid": 500}
    assert manifest["train"]["valid_plan_kind_counts"] == {
        "shortest": 250,
        "nonoptimal": 250,
    }
    assert manifest["train"]["failure_reason_counts"] == {
        "goal_not_reached": 125,
        "obstacle_collision": 125,
        "out_of_bounds": 125,
        "invalid_action_token": 125,
    }
    assert all(value == 0 for value in manifest["overlap_checks"].values())


def test_high_quality_verification_executor_labels_and_uniqueness():
    train = _read(DATA / "verification_train.jsonl")
    dev = _read(DATA / "verification_dev.jsonl")
    for rows in (train, dev):
        assert len({row["base_sample_id"] for row in rows}) == len(rows)
        assert len({(row["input_text"], row["target_text"]) for row in rows}) == len(rows)
        for row in rows:
            actual = execute_plan(row["grid"], row["candidate_plan"])
            assert actual == row["executor_label"]
            if actual["success"]:
                assert row["target_text"] == "valid"
            else:
                assert row["target_text"] == (
                    f"invalid | step={actual['first_error_step']} | "
                    f"reason={actual['reason']}"
                )


def test_high_quality_verification_has_no_generator_or_repair_overlap():
    train = _read(DATA / "verification_train.jsonl")
    dev = _read(DATA / "verification_dev.jsonl")
    generator_train = _read(
        ROOT / "artifacts/generation_trace_v9/data/transformed_train.jsonl"
    )
    generator_dev = _read(
        ROOT / "artifacts/generation_trace_v9/data/transformed_dev.jsonl"
    )
    repair_train = _read(ROOT / "data/repair_high_quality_v1/repair_train.jsonl")
    repair_dev = _read(ROOT / "data/repair_high_quality_v1/repair_dev.jsonl")

    assert not ({row["base_sample_id"] for row in train} & {
        row["base_sample_id"] for row in generator_train + repair_train
    })
    assert not ({row["base_sample_id"] for row in dev} & {
        row["base_sample_id"] for row in generator_dev + repair_dev
    })
    assert not (_semantic(train) & _semantic(generator_train + repair_train))
    assert not (_semantic(dev) & _semantic(generator_dev + repair_dev))
    assert not ({row["environment_id"] for row in train} & {
        row["environment_id"] for row in dev
    })
