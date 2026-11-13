"""Produce the five-sample Member C day-13 integration evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.mock_components import MockExecutor, MockGenerator, MockRepairer, MockVerifier
from src.pipeline import run_pipeline


BASE_GRID = {
    "rows": 3,
    "cols": 3,
    "start": [0, 0],
    "goal": [2, 2],
    "obstacles": [[0, 2]],
}
FEEDBACK_SOURCES = (
    "no_feedback",
    "self_feedback",
    "executor_feedback",
    "no_feedback",
    "executor_feedback",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/member_c_day13_mock_logs.jsonl"),
    )
    args = parser.parse_args()

    records = []
    for index, feedback_source in enumerate(FEEDBACK_SOURCES, 1):
        grid = {**BASE_GRID, "environment_id": f"c_day13_mock_{index:02d}"}
        result = run_pipeline(
            grid,
            MockGenerator(),
            MockVerifier(),
            MockRepairer(),
            MockExecutor(),
            feedback_source=feedback_source,
        )
        records.append(
            {
                "sample_id": f"c_day13_mock_{index:02d}",
                "mock": True,
                "status": "ok",
                **result,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} complete mock logs to {args.output}")


if __name__ == "__main__":
    main()
