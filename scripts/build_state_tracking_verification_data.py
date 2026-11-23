"""Add label-free state tracking to the frozen high-quality Verification data."""

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


def state_trace(grid: dict[str, Any], plan: list[str]) -> str:
    """Derive attempted coordinates without checking bounds, obstacles, or goal."""
    position = tuple(grid["start"])
    lines = [
        "Structured state tracking derived from the candidate actions only:",
        f"initial_position=({position[0]},{position[1]})",
    ]
    for step, action in enumerate(plan):
        before = position
        if action not in ACTION_DELTAS:
            lines.append(
                f"step={step} action={action} from=({before[0]},{before[1]}) attempted=unknown"
            )
            continue
        delta = ACTION_DELTAS[action]
        attempted = (before[0] + delta[0], before[1] + delta[1])
        lines.append(
            f"step={step} action={action} from=({before[0]},{before[1]}) "
            f"attempted=({attempted[0]},{attempted[1]})"
        )
        position = attempted
    lines.append(
        "Check the attempted coordinates against the grid boundary, obstacles, and goal. "
        "Use the earliest failing step."
    )
    return "\n".join(lines)


def transform(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    prompt = row["input_text"]
    if not prompt.endswith("Answer:"):
        raise ValueError(f"unexpected Verification prompt: {row['example_id']}")
    prefix = prompt[: -len("Answer:")].rstrip()
    trace = state_trace(row["grid"], row["candidate_plan"])
    result["input_text"] = f"{prefix}\n{trace}\nAnswer:"
    result["representation"] = "structured_state_tracking_v1"
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=ROOT / "data/verification_high_quality_v1"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/verification_state_tracking_v1"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {}
    hashes = {}
    max_lines = 0
    for split in ("train", "dev"):
        rows = read_jsonl(args.source / f"verification_{split}.jsonl")
        transformed = [transform(row) for row in rows]
        output = args.output / f"verification_{split}.jsonl"
        write_jsonl(output, transformed)
        counts[split] = len(transformed)
        hashes[output.name] = sha256(output)
        max_lines = max(
            max_lines,
            max(row["input_text"].count("\n") + 1 for row in transformed),
        )
    manifest = {
        "protocol": "verification-state-tracking-v1",
        "source": str(args.source.relative_to(ROOT)),
        "counts": counts,
        "sha256": hashes,
        "max_input_lines": max_lines,
        "label_free_trace": True,
        "trace_checks": [
            "coordinates are derived from start and candidate actions only",
            "trace does not consult executor labels",
            "trace does not identify validity, failure reason, or first error step",
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
