import json
from pathlib import Path

from scripts.build_state_tracking_verification_data import state_trace


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/verification_state_tracking_v1"


def _read(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_state_trace_is_coordinate_only():
    trace = state_trace(
        {"start": [1, 1]}, ["up", "left", "jump", "down"]
    )
    assert "step=0 action=up from=(1,1) attempted=(0,1)" in trace
    assert "step=1 action=left from=(0,1) attempted=(0,0)" in trace
    assert "step=2 action=jump from=(0,0) attempted=unknown" in trace
    assert "step=3 action=down from=(0,0) attempted=(1,0)" in trace
    assert "obstacle_collision" not in trace
    assert "out_of_bounds" not in trace
    assert "goal_not_reached" not in trace


def test_state_tracking_preserves_rows_and_targets():
    for split in ("train", "dev"):
        source = _read(ROOT / f"data/verification_high_quality_v1/verification_{split}.jsonl")
        transformed = _read(DATA / f"verification_{split}.jsonl")
        assert len(source) == len(transformed)
        for before, after in zip(source, transformed):
            assert before["example_id"] == after["example_id"]
            assert before["target_text"] == after["target_text"]
            assert before["candidate_plan"] == after["candidate_plan"]
            assert after["representation"] == "structured_state_tracking_v1"
            assert after["input_text"].endswith("Answer:")
