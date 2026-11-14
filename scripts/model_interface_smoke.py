"""Shared smoke runner; all four legacy command entry points use this API."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import platform
from pathlib import Path

from src.models.api import GridModel
from src.models.backend import ModelConfig
from src.models.prompts import PROTOCOL_VERSION


GRID = {
    "environment_id": "toy_grid_0001", "rows": 3, "cols": 3,
    "start": [0, 0], "goal": [2, 2], "obstacles": [[0, 2]],
}
CANDIDATES = [
    {"case_id": "candidate_01", "plan": ["right", "down", "down", "right"]},
    {"case_id": "candidate_02", "plan": ["right", "right", "down"]},
]

# Hand-designed smoke fixture, not official PPNL data or a benchmark split.
GRID_6X6 = {
    "environment_id": "toy_grid_6x6_001", "rows": 6, "cols": 6,
    "start": [0, 0], "goal": [5, 5],
    "obstacles": [[0, 2], [1, 2], [3, 4]],
}
CANDIDATES_6X6 = [
    # Down the first column, then right along the bottom row: reaches [5, 5].
    {"case_id": "candidate_01", "plan": ["down"] * 5 + ["right"] * 5},
    # The second right enters [0, 2]: obstacle_collision at zero-based step=1.
    {"case_id": "candidate_02", "plan": ["right", "right", "down"]},
]


def main(default_task="both", default_checkpoint="google-t5/t5-small"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("generation", "verification", "both"), default=default_task)
    parser.add_argument("--config", type=Path, help="JSON config; CLI checkpoint overrides its value")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--grid-size", type=int, choices=(3, 6), default=3)
    args = parser.parse_args()
    grid = GRID_6X6 if args.grid_size == 6 else GRID
    candidates = CANDIDATES_6X6 if args.grid_size == 6 else CANDIDATES
    values = {"checkpoint": default_checkpoint}
    if args.config:
        with args.config.open(encoding="utf-8") as handle:
            values.update(json.load(handle))
    if args.checkpoint:
        values["checkpoint"] = args.checkpoint
    config = ModelConfig(**values)

    started_at = datetime.now(timezone.utc)
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    # Avoid embedding a user-supplied checkpoint path in the result filename.
    output_path = results_dir / (
        f"{args.task}_smoke_{args.grid_size}x{args.grid_size}_"
        + started_at.strftime("%Y%m%dT%H%M%S_%fZ") + ".json"
    )
    report = {
        "schema_version": 2,
        "started_at_utc": started_at.isoformat(),
        "task": args.task,
        "checkpoint": config.checkpoint,
        "config": asdict(config),
        "protocol_version": PROTOCOL_VERSION,
        "step_index_base": 0,
        "goal_not_reached_step_convention": "len(plan)",
        "grid": grid,
        "fixture_source": "hand_designed_smoke_example_not_official_dataset",
        "cases": [],
        "status": "started",
    }
    try:
        import torch
        import transformers

        report["versions"] = {
            "python": platform.python_version(),
            "torch": torch.__version__, "transformers": transformers.__version__,
        }
        print("Checkpoint:", config.checkpoint)
        print("Device:", config.device)
        print("Protocol:", PROTOCOL_VERSION)
        print(f"Grid: {grid['rows']}x{grid['cols']} ({grid['environment_id']})")
        client = GridModel(config)
        print("Loading tokenizer and model...")
        client.backend.load()
        print("Model loaded successfully")

        jobs = []
        if args.task in ("generation", "both"):
            jobs.append({"task": "generation", "case_id": "generation_01"})
        if args.task in ("verification", "both"):
            jobs.extend({"task": "verification", **candidate} for candidate in candidates)

        for job in jobs:
            record = {**job, "status": "started"}
            report["cases"].append(record)
            if job["task"] == "generation":
                result = client.generate_plan(grid)
            else:
                result = client.verify_plan(grid, job["plan"])
            record.update(result=result, status="inference_completed")
            metadata = result["metadata"]
            print(f"\n--- {job['case_id']} ---")
            if "plan" in job:
                print("Candidate plan:", job["plan"])
            print("Input tokens:", metadata["input_tokens"])
            print("Input truncated:", metadata["input_truncated"])
            print("Raw output:", repr(result["raw_output"]))
            print("Parsed result:", {k: v for k, v in result.items() if k not in ("metadata", "raw_output")})
            print("Generated tokens:", metadata["output_tokens_including_eos"])
            print("Ended with EOS:", metadata["ended_with_eos"])
            print("Reached output limit:", metadata["reached_output_limit"])
            print(f"Inference time: {metadata['inference_seconds']:.2f} seconds")

        report["status"] = "inference_completed"
        report["parse_success_count"] = sum(
            record["result"]["parse_success"] for record in report["cases"]
        )
        print("\nParsed cases:", report["parse_success_count"], "/", len(jobs))
        print("This is a smoke test, not a model accuracy measurement.")
    except Exception as exc:
        report["status"] = "error"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if report["cases"] and report["cases"][-1]["status"] == "started":
            report["cases"][-1].update(status="error", error=report["error"])
        raise
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        with output_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print("Results saved to:", output_path)


if __name__ == "__main__":
    main()
