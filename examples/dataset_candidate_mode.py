"""Offline demonstration of paired methods using a frozen dataset candidate.

This synthetic example does not read or modify held-out data and makes no API
calls. Replace the adapters with the selected Generator/Verifier/Repair/Executor
implementations for a real run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import run_pipeline_comparison
from src.smoke_integration import (
    DeterministicRepairPlaceholder,
    ExecutorVerifierAdapter,
    RealExecutorAdapter,
)


class GeneratorMustNotRun:
    def generate(self, grid):  # pragma: no cover - failure guard
        raise AssertionError("Dataset mode must skip Generator")


GRID = {
    "environment_id": "synthetic-env-001",
    "rows": 3,
    "cols": 3,
    "start": [0, 0],
    "goal": [0, 2],
    "obstacles": [],
}

FROZEN_ROW = {
    "sample_id": "synthetic-id-001",
    "base_sample_id": "synthetic-base-001",
    "environment_id": GRID["environment_id"],
    "grid": GRID,
    "candidate_plan": ["right"],
}

result = run_pipeline_comparison(
    GRID,
    GeneratorMustNotRun(),
    ExecutorVerifierAdapter(),
    DeterministicRepairPlaceholder(["right", "right"]),
    RealExecutorAdapter(),
    candidate_source="dataset",
    candidate_record=FROZEN_ROW,
)

print(
    json.dumps(
        {
            "sample_id": result["sample_id"],
            "base_sample_id": result["base_sample_id"],
            "candidate_plan": result["candidate_plan"],
            "candidate_plan_sha256": result["candidate_plan_sha256"],
            "generator_called": result["generation"] is not None,
            "method_success": {
                name: branch["final_result"]["success"]
                for name, branch in result["branches"].items()
            },
        },
        ensure_ascii=False,
        indent=2,
    )
)
