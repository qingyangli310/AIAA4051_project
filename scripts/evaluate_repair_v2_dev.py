"""Test Repair v2 on its archived Dev split using the pipeline adapter."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_adapters import Seq2SeqRepairerAdapter, TransformersTextGenerator
from src.smoke_integration import RealExecutorAdapter


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.limit <= 50:
        raise ValueError("--limit must be between 1 and 50")
    weights = args.checkpoint / "model.safetensors"
    with weights.open("rb") as stream:
        if stream.read(100).startswith(b"version https://git-lfs.github.com/spec/"):
            raise RuntimeError("Checkpoint is a Git LFS pointer. Download the Repair v2 weights with git lfs pull first.")
    data_path = ROOT / "data/repair_high_quality_v2/repair_dev.jsonl"
    archive_path = ROOT / "results/repair_flan_t5_base_high_quality_v2/dev_predictions.jsonl"
    rows = read_jsonl(data_path)
    archive = {row["example_id"]: row for row in read_jsonl(archive_path)}
    if len(rows) != 50 or {row["example_id"] for row in rows} != set(archive):
        raise ValueError("Repair v2 Dev50 does not match the archived sample IDs")
    args.output.mkdir(parents=True, exist_ok=False)

    print("Loading Repair v2 on CPU...", flush=True)
    runner = TransformersTextGenerator(args.checkpoint, device="cpu")
    repairer = Seq2SeqRepairerAdapter(runner, output_format="trace_only", max_new_tokens=128)
    executor = RealExecutorAdapter()
    records = []
    started = time.perf_counter()
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows[:args.limit], start=1):
            record = {"example_id": row["example_id"], "feedback_source": row["feedback_source"]}
            try:
                def infer(prompt: str, limit: int) -> str:
                    if prompt != row["input_text"]:
                        raise ValueError("Runtime prompt differs from archived Repair v2 input")
                    return runner(prompt, limit)

                repairer.infer_text = infer
                result = repairer.repair(
                    environment_id=row["environment_id"],
                    grid=row["grid"],
                    candidate_plan=row["candidate_plan"],
                    feedback_source=row["feedback_source"],
                    verifier_feedback=row["verifier_feedback"],
                )
                execution = executor.execute(row["grid"], result["repaired_plan"]) if result["parse_success"] else None
                record.update(
                    status="ok", repair=result, executor=execution,
                    success=bool(execution and execution["success"]),
                    raw_output_matches_archive=result["raw_output"] == archive[row["example_id"]]["raw_output"],
                )
                print(f"[{index}/{args.limit}] raw_output={result['raw_output']}", flush=True)
                print(f"parse={result['parse_success']} success={record['success']} actions={result['repaired_plan']}", flush=True)
            except Exception as exc:
                record.update(status="error", success=False, error=f"{type(exc).__name__}: {exc}")
                print(f"[{index}/{args.limit}] ERROR {record['error']}", flush=True)
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()

    report = {
        "scope": "repair_v2_dev_cpu",
        "checkpoint": str(args.checkpoint.resolve()),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "examples": len(records),
        "success_count": sum(row["success"] for row in records),
        "success_rate": sum(row["success"] for row in records) / len(records),
        "parse_success_count": sum(row.get("repair", {}).get("parse_success", False) for row in records),
        "archived_raw_output_match_count": sum(row.get("raw_output_matches_archive", False) for row in records),
        "error_count": sum(row["status"] == "error" for row in records),
        "feedback_counts": dict(Counter(row["feedback_source"] for row in records)),
        "inference_and_scoring_seconds": time.perf_counter() - started,
        "decoding": {"max_new_tokens": 128, "num_beams": 1, "do_sample": False},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 1 if report["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
