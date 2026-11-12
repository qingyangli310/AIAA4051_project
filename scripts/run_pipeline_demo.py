"""Run the member-C integration demo for the 12 November meeting."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.mock_components import MockExecutor, MockGenerator, MockRepairer, MockVerifier
from src.pipeline import run_pipeline


TOY_GRID = {
    "environment_id": "grid_0001",
    "rows": 3,
    "cols": 3,
    "start": [0, 0],
    "goal": [2, 2],
    "obstacles": [[0, 2]],
}


def main() -> None:
    result = run_pipeline(
        grid=TOY_GRID,
        generator=MockGenerator(),
        verifier=MockVerifier(),
        repairer=MockRepairer(),
        executor=MockExecutor(),
    )
    feedback = result["verification"]

    print(f"Environment: {result['environment_id']}")
    print(f"Generated plan: {', '.join(result['candidate_plan'])}")
    print(
        "Verification: "
        f"invalid | step={feedback['first_error_step']} | reason={feedback['reason']}"
    )
    print(f"Repaired plan: {', '.join(result['repaired_plan'])}")
    status = "success" if result["final_result"]["success"] else "failure"
    print(f"Final executor result: {status}")


if __name__ == "__main__":
    main()
