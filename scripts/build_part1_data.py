"""Build all frozen Part 1 train, Dev, ID, and OOD task files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.part1.data import build_part1_datasets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "data/part1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build_part1_datasets(args.root, args.output, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
