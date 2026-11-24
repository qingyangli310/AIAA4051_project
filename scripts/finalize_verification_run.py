#!/usr/bin/env python3
"""Materialize the selected Verification run report from saved predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.output_parser import parse_verification


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/verification_high_quality_v1",
    )
    args = parser.parse_args()

    raw_report = read_json(args.source_dir / "run_report.raw.json")
    scoring = read_json(args.source_dir / "dev_scoring.normalized.json")
    prediction_lines = (args.source_dir / "dev_predictions.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    normalized_predictions = []
    for line in prediction_lines:
        row = json.loads(line)
        verdict = row["target_text"].rsplit("Verdict:", 1)[-1].strip()
        parsed = parse_verification(row["raw_output"])
        normalized_predictions.append(
            {
                **row,
                "verdict_text": verdict,
                "normalized_output": parsed.get("normalized_output"),
                "parse_success": parsed["parse_success"],
            }
        )

    final_report = {
        **raw_report,
        "checkpoint": "artifacts/verification_trace_v1/checkpoint-final",
        "checkpoint_sha256": "c125484941f4744540cfd250ade8ca1d921f0efb8d6a88fa22a03cc083a474f1",
        "dev_scoring": scoring,
        "scoring_correction": {
            "predictions_changed": False,
            "retraining_performed": False,
            "reason": (
                "T5 converted the newline before the explicit Verdict label to a space; "
                "the final parser accepts both whitespace forms."
            ),
            "raw_training_time_report": "run_report.raw.json",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "run_report.json", final_report)
    write_json(args.output_dir / "dev_scoring.json", scoring)
    write_json(args.output_dir / "run_report.raw.json", raw_report)
    with (args.output_dir / "dev_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in normalized_predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
