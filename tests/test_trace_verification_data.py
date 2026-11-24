import json
from pathlib import Path

from scripts.build_trace_verification_data import trace_target
from src.models.output_parser import parse_verification


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/verification_trace_v1"


def _read(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_trace_target_and_final_verdict_parser():
    row = {
        "candidate_plan": ["right", "down"],
        "target_text": "invalid | step=1 | reason=obstacle_collision",
        "executor_label": {
            "success": False,
            "reason": "obstacle_collision",
            "first_error_step": 1,
            "executed_steps": 1,
            "trace": [[0, 0], [0, 1]],
        },
    }
    target = trace_target(row)
    assert target == (
        "Trace: (0,0) right (0,1) attempt down (1,1)\n"
        "Verdict: invalid | step=1 | reason=obstacle_collision"
    )
    parsed = parse_verification(target, 2)
    assert parsed["parse_success"] is True
    assert parsed["valid"] is False
    assert parsed["first_error_step"] == 1
    assert parsed["reason"] == "obstacle_collision"
    same_line = target.replace("\nVerdict:", " Verdict:")
    assert parse_verification(same_line, 2)["parse_success"] is True


def test_trace_dataset_preserves_strict_verdicts():
    for split in ("train", "dev"):
        rows = _read(DATA / f"verification_{split}.jsonl")
        for row in rows:
            assert row["target_text"].endswith(row["verdict_text"])
            parsed = parse_verification(row["target_text"], len(row["candidate_plan"]))
            assert parsed["parse_success"] is True
            assert parsed["normalized_output"] == row["verdict_text"]
