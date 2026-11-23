"""Build a trace-supervised Verification representation from the frozen rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.executor.executor import ACTION_DELTAS
from src.part1.data import read_jsonl, write_jsonl


def trace_target(row: dict[str, Any]) -> str:
    label = row["executor_label"]
    plan = row["candidate_plan"]
    trace = label["trace"]
    pieces = [f"({trace[0][0]},{trace[0][1]})"]
    executed = int(label["executed_steps"])
    for step in range(executed):
        position = trace[step + 1]
        pieces.extend([plan[step], f"({position[0]},{position[1]})"])
    if not label["success"] and label["reason"] != "goal_not_reached":
        step = int(label["first_error_step"])
        action = plan[step]
        before = tuple(trace[-1])
        if action in ACTION_DELTAS:
            delta = ACTION_DELTAS[action]
            attempted = f"({before[0] + delta[0]},{before[1] + delta[1]})"
        else:
            attempted = "unknown"
        pieces.extend(["attempt", action, attempted])
    return f"Trace: {' '.join(pieces)}\nVerdict: {row['target_text']}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=ROOT / "data/verification_high_quality_v1"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/verification_trace_v1"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {}
    hashes = {}
    for split in ("train", "dev"):
        rows = read_jsonl(args.source / f"verification_{split}.jsonl")
        transformed = []
        for row in rows:
            result = dict(row)
            result["verdict_text"] = row["target_text"]
            result["target_text"] = trace_target(row)
            result["representation"] = "trace_then_verdict_v1"
            transformed.append(result)
        output = args.output / f"verification_{split}.jsonl"
        write_jsonl(output, transformed)
        counts[split] = len(transformed)
        hashes[output.name] = sha256(output)
    manifest = {
        "protocol": "verification-trace-v1",
        "source": str(args.source.relative_to(ROOT)),
        "counts": counts,
        "sha256": hashes,
        "target_schema": "Trace: <state transitions>\\nVerdict: <strict verification output>",
        "final_verdict_parser": "final explicitly labelled Verdict line only",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
