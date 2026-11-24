from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.data.candidate_source import load_candidate_records
from src.pipeline import run_pipeline, run_pipeline_batch, run_pipeline_comparison, run_pipeline_comparison_batch
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
    }


class DatasetCandidateModeTests(unittest.TestCase):
    def test_dataset_mode_skips_generator_and_preserves_ids(self):
        result = run_pipeline(
            GRID,
            GeneratorMustNotRun(),
            ExecutorVerifierAdapter(),
            DeterministicRepairPlaceholder(["right", "right"]),
            RealExecutorAdapter(),
            candidate_source="dataset",
            candidate_record=candidate_row(),
        )
        self.assertIsNone(result["generation"])
        self.assertEqual(result["candidate_source"], "dataset")
        self.assertEqual(result["candidate_plan"], ["right"])
        self.assertEqual(result["sample_id"], "sample-1")
        self.assertEqual(result["base_sample_id"], "base-1")

    def test_generator_mode_keeps_existing_behavior(self):
        generator = CountingGenerator()
        result = run_pipeline(
            GRID,
            generator,
            ExecutorVerifierAdapter(),
            DeterministicRepairPlaceholder(["right", "right"]),
            RealExecutorAdapter(),
        )
        self.assertEqual(generator.calls, 1)
        self.assertEqual(result["candidate_source"], "generator")
        self.assertEqual(result["generation"]["plan"], ["right", "right"])

    def test_three_methods_receive_identical_dataset_candidate(self):
        candidate = ["right"]
        verifier = CapturingVerifier()
        repairer = CapturingRepairer()
        result = run_pipeline_comparison(
            GRID,
            GeneratorMustNotRun(),
            verifier,
            repairer,
            RealExecutorAdapter(),
            candidate_source="dataset",
            candidate_record=candidate_row(candidate),
        )
        self.assertEqual(result["candidate_plan"], candidate)
        self.assertEqual(result["branches"]["direct"]["repaired_plan"], candidate)
        self.assertEqual(verifier.plans, [candidate])
        self.assertEqual([r["candidate_plan"] for r in repairer.requests], [candidate, candidate])
        self.assertEqual(len(result["candidate_plan_sha256"]), 64)

    def test_id_ood_loader_reads_candidate_aliases_and_preserves_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            id_path, ood_path = Path(temp) / "id.jsonl", Path(temp) / "ood.jsonl"
            id_record = {
                **candidate_row(["left"]),
                "candidate_plan": None,
                "corrupted_plan": ["left"],
                "corruption_type": "action_substitution",
                "original_plan": ["right", "right"],
            }
            ood_record = {
                **candidate_row(["up", "left"]),
                "candidate_plan": None,
                "corrupted_plan": ["up", "left"],
                "corruption_type": ["action_insertion", "action_substitution"],
                "original_plan": ["right", "right"],
            }
            id_path.write_text(json.dumps(id_record) + "\n", encoding="utf-8")
            ood_path.write_text(json.dumps(ood_record) + "\n", encoding="utf-8")
            id_row = load_candidate_records(id_path, condition="id_single_error")[0]
            ood_row = load_candidate_records(ood_path, condition="ood_double_error")[0]
        self.assertEqual(id_row["candidate_plan"], ["left"])
        self.assertEqual((id_row["sample_id"], id_row["base_sample_id"]), ("sample-1", "base-1"))
        self.assertEqual(id_row["condition"], "id_single_error")
        self.assertEqual(ood_row["candidate_plan"], ["up", "left"])
        self.assertEqual(ood_row["condition"], "ood_double_error")
        self.assertNotIn("original_plan", id_row)
        self.assertNotIn("original_plan", ood_row)

    def test_comparison_batch_maps_results_to_source_ids(self):
        result = run_pipeline_comparison_batch(
            [candidate_row()],
            GeneratorMustNotRun(),
            ExecutorVerifierAdapter(),
            DeterministicRepairPlaceholder(["right", "right"]),
            RealExecutorAdapter(),
        )[0]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            (result["sample_id"], result["base_sample_id"], result["environment_id"]),
            ("sample-1", "base-1", "env-1"),
        )
        self.assertEqual(set(result["branches"]), {"direct", "self_feedback", "executor_feedback"})

    def test_loader_does_not_modify_source_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "frozen.jsonl"
            path.write_text(
                json.dumps(
                    {
                        **candidate_row(),
                        "corruption_type": "action_substitution",
                        "original_plan": ["right", "right"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            load_candidate_records(path, condition="id_single_error")
            after = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(before, after)

    def test_dataset_mode_rejects_missing_candidate_and_misaligned_batch(self):
        with self.assertRaisesRegex(ValueError, "candidate_plan or corrupted_plan"):
            run_pipeline(
                GRID,
                GeneratorMustNotRun(),
                ExecutorVerifierAdapter(),
                DeterministicRepairPlaceholder(["right", "right"]),
                RealExecutorAdapter(),
                candidate_source="dataset",
                candidate_record={"sample_id": "x"},
            )
        with self.assertRaisesRegex(ValueError, "one candidate record per grid"):
            run_pipeline_batch(
                [GRID],
                None,
                ExecutorVerifierAdapter(),
                DeterministicRepairPlaceholder(["right"]),
                RealExecutorAdapter(),
                candidate_source="dataset",
                candidate_records=[],
            )


if __name__ == "__main__":
    unittest.main()
