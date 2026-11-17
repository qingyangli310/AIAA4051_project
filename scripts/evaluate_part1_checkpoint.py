"""Evaluate a saved Part 1 checkpoint on one frozen JSONL task file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.part1.data import read_jsonl
from src.part1.evaluate import score_predictions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("generation", "verification", "repair"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-output-length", type=int, default=128)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install requirements-project6.txt in the frozen environment") from exc
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("formal evaluation requires CUDA; use --allow-cpu only for diagnostics")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    rows = read_jsonl(args.data)
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(args.checkpoint), local_files_only=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    raw_outputs, predictions = [], []
    started = time.perf_counter()
    for row in rows:
        inputs = tokenizer(row["input_text"], return_tensors="pt", truncation=False).to(device)
        with torch.inference_mode():
            token_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_output_length,
                do_sample=False,
                num_beams=1,
            )[0]
        raw = tokenizer.decode(token_ids, skip_special_tokens=True)
        raw_outputs.append(raw)
        predictions.append(
            {
                "example_id": row["example_id"],
                "environment_id": row["environment_id"],
                "target_text": row["target_text"],
                "raw_output": raw,
            }
        )
    elapsed = time.perf_counter() - started
    args.output.mkdir(parents=True)
    (args.output / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    report = {
        "task": args.task,
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "device": device,
        "examples": len(rows),
        "inference_seconds": elapsed,
        "scoring": score_predictions(args.task, rows, raw_outputs),
        "decoding": {"do_sample": False, "num_beams": 1},
    }
    (args.output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
