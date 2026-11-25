from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.data.candidate_source import load_candidate_records
from src.pipeline import (
    run_pipeline,
    run_pipeline_batch,
    run_pipeline_comparison,
    run_pipeline_comparison_batch,
)
from src.smoke_integration import (
    DeterministicRepairPlaceholder,
    ExecutorVerifierAdapter,
    RealExecutorAdapter,
)


GRID = {
    "environment_id": "env-1",
    "rows": 3,
    "cols": 3,
    "start": [0, 0],
    "goal": [0, 2],
    "obstacles": [],
}


class GeneratorMustNotRun:
    def generate(self, grid):
        raise AssertionError("Generator was called in dataset mode")


class CountingGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, grid):
        self.calls += 1
        return {"plan": ["right", "right"], "parse_success": True}


class CapturingVerifier:
    def __init__(self):
        self.plans = []

    def verify(self, grid, plan):
        self.plans.append(list(plan))
        return ExecutorVerifierAdapter().verify(grid, plan)


class CapturingRepairer:
    def __init__(self):
        self.requests = []

    def repair(self, **request):
        self.requests.append(request)
        return {
            "raw_output": "right, right",
            "repaired_plan": ["right", "right"],
            "parse_success": True,
        }


def candidate_row(plan=None):
    return {
        "sample_id": "sample-1",
        "base_sample_id": "base-1",
        "environment_id": "env-1",
        "grid": GRID,
        "candidate_plan": list(plan if plan is not None else ["right"]),
        "corruption_type": "premature_stopping",
        "optimal_length": 2,
    }


def test_dataset_mode_skips_generator_and_preserves_ids():
    result = run_pipeline(
        GRID,
        GeneratorMustNotRun(),
        ExecutorVerifierAdapter(),
        DeterministicRepairPlaceholder(["right", "right"]),
        RealExecutorAdapter(),
        candidate_source="dataset",
        candidate_record=candidate_row(),
    )
    assert result["generation"] is None
    assert result["candidate_source"] == "dataset"
    assert result["candidate_plan"] == ["right"]
    assert result["sample_id"] == "sample-1"
    assert result["base_sample_id"] == "base-1"


def test_generator_mode_keeps_existing_behavior():
    generator = CountingGenerator()
    result = run_pipeline(
        GRID,
        generator,
        ExecutorVerifierAdapter(),
        DeterministicRepairPlaceholder(["right", "right"]),
        RealExecutorAdapter(),
    )
    assert generator.calls == 1
    assert result["candidate_source"] == "generator"
    assert result["generation"]["plan"] == ["right", "right"]


def test_all_conditions_share_one_dataset_candidate():
    candidate = ["right"]
    verifier = CapturingVerifier()
    repairer = CapturingRepairer()
    result = run_pipeline_comparison(
        GRID,
        GeneratorMustNotRun(),
        verifier,
        repairer,
        RealExecutorAdapter(),
        conditions=("direct", "no_feedback", "self_feedback", "executor_feedback"),
        candidate_source="dataset",
        candidate_record=candidate_row(candidate),
    )
    assert result["candidate_plan"] == candidate
    assert result["branches"]["direct"]["repaired_plan"] == candidate
    assert verifier.plans == [candidate]
    assert [request["candidate_plan"] for request in repairer.requests] == [
        candidate,
        candidate,
        candidate,
    ]
    assert len(result["candidate_plan_sha256"]) == 64


def test_loader_reads_aliases_sanitizes_targets_and_preserves_optimality(tmp_path: Path):
    source = tmp_path / "id.jsonl"
    record = {
        **candidate_row(["left"]),
        "candidate_plan": None,
        "corrupted_plan": ["left"],
        "corruption_type": "action_substitution",
        "target_plan": ["right", "right"],
        "target_text": "right, right",
    }
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    loaded = load_candidate_records(source, condition="id_single_error")[0]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert loaded["candidate_plan"] == ["left"]
    assert loaded["optimal_length"] == 2
    assert "target_plan" not in loaded
    assert "target_text" not in loaded


def test_ood_loader_requires_two_corruptions(tmp_path: Path):
    source = tmp_path / "ood.jsonl"
    source.write_text(
        json.dumps(
            {
                **candidate_row(["up", "left"]),
                "corruption_type": ["action_insertion", "action_substitution"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_candidate_records(source, condition="ood_double_error")[0]
    assert loaded["candidate_plan"] == ["up", "left"]
    assert loaded["condition"] == "ood_double_error"


def test_comparison_batch_supports_shuffled_feedback_in_dataset_mode():
    first = candidate_row(["right"])
    second = {
        **candidate_row(["up"]),
        "sample_id": "sample-2",
        "base_sample_id": "base-2",
    }
    records = run_pipeline_comparison_batch(
        [first, second],
        GeneratorMustNotRun(),
        ExecutorVerifierAdapter(),
        DeterministicRepairPlaceholder(["right", "right"]),
        RealExecutorAdapter(),
        conditions=(
            "direct",
            "no_feedback",
            "self_feedback",
            "executor_feedback",
            "shuffled_executor_feedback",
        ),
        candidate_source="dataset",
    )
    assert all(record["status"] == "ok" for record in records)
    assert all(record["generation"] is None for record in records)
    assert all("shuffled_executor_feedback" in record["branches"] for record in records)
    assert [record["sample_id"] for record in records] == ["sample-1", "sample-2"]


def test_dataset_mode_rejects_missing_candidate_and_misaligned_batch():
    with pytest.raises(ValueError, match="candidate_plan or corrupted_plan"):
        run_pipeline(
            GRID,
            GeneratorMustNotRun(),
            ExecutorVerifierAdapter(),
            DeterministicRepairPlaceholder(["right", "right"]),
            RealExecutorAdapter(),
            candidate_source="dataset",
            candidate_record={"sample_id": "x"},
        )
    with pytest.raises(ValueError, match="one candidate record per grid"):
        run_pipeline_batch(
            [GRID],
            None,
            ExecutorVerifierAdapter(),
            DeterministicRepairPlaceholder(["right"]),
            RealExecutorAdapter(),
            candidate_source="dataset",
            candidate_records=[],
        )
