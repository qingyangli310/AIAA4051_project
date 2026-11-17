"""Train, reload, predict, and diagnose one formal Part 1 seq2seq run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.part1.data import read_jsonl
from src.part1.evaluate import score_predictions


MODELS = {
    "t5-small": (
        "google-t5/t5-small",
        "df1b051c49625cf57a3d0d8d3863ed4d13564fe4",
    ),
    "bart-base": (
        "facebook/bart-base",
        "aadd2ab0ae0c8268c7c9693540e9904811f36177",
    ),
}
PLAN_ACTIONS = ("up", "down", "left", "right")


def diagnostic_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Choose an action-balanced, easy-to-hard set for an overfit probe."""
    groups = {
        action: sorted(
            (row for row in rows if row["target_plan"][0] == action),
            key=lambda row: (len(row["target_plan"]), row["example_id"]),
        )
        for action in PLAN_ACTIONS
    }
    selected = []
    index = 0
    while len(selected) < count:
        progressed = False
        for action in PLAN_ACTIONS:
            if index < len(groups[action]):
                selected.append(groups[action][index])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise ValueError(f"could not select {count} diagnostic rows")
        index += 1
    return selected


def parameter_signature(model: Any) -> dict[str, float]:
    names = (
        "shared.weight",
        "encoder.embed_tokens.weight",
        "model.shared.weight",
        "model.encoder.embed_tokens.weight",
    )
    parameters = dict(model.named_parameters())
    name = next((candidate for candidate in names if candidate in parameters), None)
    if name is None:
        name = next(iter(parameters))
    tensor = parameters[name].detach().float()
    return {
        "parameter": name,
        "sum": tensor.sum().item(),
        "norm": tensor.norm().item(),
    }


def encode_rows(rows: list[dict[str, Any]], tokenizer: Any, max_input: int, max_output: int):
    encoded = []
    lengths = {"max_input_tokens": 0, "max_target_tokens": 0}
    for index, row in enumerate(rows, 1):
        source = tokenizer(row["input_text"], truncation=False)
        target = tokenizer(text_target=row["target_text"], truncation=False)
        input_count = len(source["input_ids"])
        target_count = len(target["input_ids"])
        if input_count > max_input or target_count > max_output:
            raise ValueError(
                f"row {index} exceeds token limits: input={input_count}, target={target_count}"
            )
        lengths["max_input_tokens"] = max(lengths["max_input_tokens"], input_count)
        lengths["max_target_tokens"] = max(lengths["max_target_tokens"], target_count)
        encoded.append(
            {
                "input_ids": source["input_ids"],
                "attention_mask": source["attention_mask"],
                "labels": target["input_ids"],
            }
        )
    return encoded, lengths


def teacher_forced_loss(model: Any, tokenizer: Any, row: dict[str, Any], device: str) -> float:
    inputs = tokenizer(row["input_text"], return_tensors="pt", truncation=False).to(device)
    labels = tokenizer(
        text_target=row["target_text"], return_tensors="pt", truncation=False
    )["input_ids"].to(device)
    return model(**inputs, labels=labels).loss.item()


def conditioning_probe(model: Any, tokenizer: Any, rows: list[dict[str, Any]], device: str) -> dict[str, Any]:
    import torch

    matched, mismatched = [], []
    model.eval()
    with torch.inference_mode():
        for index, row in enumerate(rows):
            matched.append(teacher_forced_loss(model, tokenizer, row, device))
            wrong = {**row, "input_text": rows[(index + 1) % len(rows)]["input_text"]}
            mismatched.append(teacher_forced_loss(model, tokenizer, wrong, device))
    return {
        "matched_loss": sum(matched) / len(matched),
        "mismatched_loss": sum(mismatched) / len(mismatched),
        "mismatched_minus_matched": sum(mismatched) / len(mismatched) - sum(matched) / len(matched),
        "matched_better_count": sum(a < b for a, b in zip(matched, mismatched)),
        "examples": len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("generation", "verification", "repair"), required=True)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/part1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--diagnostic-train-examples", type=int)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-output-length", type=int, default=128)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    diagnostic = args.diagnostic_train_examples is not None
    if not diagnostic and not 1 <= args.epochs <= 3:
        raise ValueError("formal Part 1 runs allow one to three epochs")
    if diagnostic and not 2 <= args.diagnostic_train_examples <= 100:
        raise ValueError("diagnostic overfit probes require 2 to 100 examples")
    if diagnostic and not 1 <= args.epochs <= 100:
        raise ValueError("diagnostic overfit probes allow one to 100 epochs")
    # The three Part 1 datasets are deliberately small.  A large effective
    # batch (8 * 2) left Verification and Repair with only 282 optimiser
    # updates and produced majority-like predictions despite balanced labels.
    # Use the same update-dense defaults that pass the Generation conditioning
    # probe; callers can still override every value on the command line.
    if args.learning_rate is None:
        args.learning_rate = 2e-4
    if args.batch_size is None:
        args.batch_size = 4
    if args.gradient_accumulation is None:
        args.gradient_accumulation = 1
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    try:
        import torch
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError("install requirements-project6.txt in the frozen GPU environment") from exc
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("formal training requires CUDA; pass --allow-cpu only for diagnostics")

    train_path = args.data_dir / f"{args.task}_train.jsonl"
    dev_path = args.data_dir / f"{args.task}_dev.jsonl"
    train_rows, dev_rows = read_jsonl(train_path), read_jsonl(dev_path)
    if diagnostic:
        train_rows = diagnostic_rows(train_rows, args.diagnostic_train_examples)
        dev_rows = list(train_rows)
    checkpoint_name, revision = MODELS[args.model]
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_name, revision=revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_name, revision=revision)
    before = parameter_signature(model)
    encoded_train, train_lengths = encode_rows(
        train_rows, tokenizer, args.max_input_length, args.max_output_length
    )
    encoded_dev, dev_lengths = encode_rows(
        dev_rows, tokenizer, args.max_input_length, args.max_output_length
    )
    args.output.mkdir(parents=True)
    checkpoint_dir = args.output / "checkpoint-final"
    training_args = TrainingArguments(
        output_dir=str(args.output / "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        warmup_steps=0,
        weight_decay=0.01,
        optim="adamw_torch",
        bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        seed=42,
        data_seed=42,
        logging_steps=1,
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
    )
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    preview_batch = collator(encoded_train[: min(args.batch_size, len(encoded_train))])
    preview_lengths = {
        len(row["labels"]) for row in encoded_train[: min(args.batch_size, len(encoded_train))]
    }
    if len(preview_lengths) > 1 and not (preview_batch["labels"] == -100).any():
        raise AssertionError("expected padded labels to contain -100")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train,
        eval_dataset=encoded_dev,
        data_collator=collator,
        processing_class=tokenizer,
    )
    started = time.perf_counter()
    train_result = trainer.train()
    duration = time.perf_counter() - started
    after = parameter_signature(model)
    if before == after:
        raise AssertionError("selected model parameter signature did not change")
    dev_metrics = trainer.evaluate()
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    history = trainer.state.log_history
    completed_steps = trainer.state.global_step
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reloaded = AutoModelForSeq2SeqLM.from_pretrained(str(checkpoint_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reloaded.to(device).eval()
    raw_outputs = []
    prediction_rows = []
    for row in dev_rows:
        inputs = tokenizer(row["input_text"], return_tensors="pt", truncation=False).to(device)
        with torch.inference_mode():
            tokens = reloaded.generate(
                **inputs,
                max_new_tokens=args.max_output_length,
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=0,
                early_stopping=False,
            )[0]
        raw = tokenizer.decode(tokens, skip_special_tokens=True)
        raw_outputs.append(raw)
        prediction_rows.append(
            {
                "example_id": row["example_id"],
                "environment_id": row["environment_id"],
                "target_text": row["target_text"],
                "raw_output": raw,
            }
        )
    probe = conditioning_probe(reloaded, tokenizer, dev_rows, device)
    scoring = score_predictions(args.task, dev_rows, raw_outputs)
    (args.output / "dev_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    report = {
        "formal_part1_run": not diagnostic,
        "diagnostic_overfit_probe": diagnostic,
        "task": args.task,
        "model_key": args.model,
        "model_name": checkpoint_name,
        "model_revision": revision,
        "checkpoint": str(checkpoint_dir),
        "seed": 42,
        "train_count": len(train_rows),
        "dev_count": len(dev_rows),
        "epochs": args.epochs,
        "completed_steps": completed_steps,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "device": device,
        "bf16": training_args.bf16,
        "train_seconds": duration,
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "dev_sha256": hashlib.sha256(dev_path.read_bytes()).hexdigest(),
        "train_lengths": train_lengths,
        "dev_lengths": dev_lengths,
        "parameter_signature_before": before,
        "parameter_signature_after": after,
        "train_metrics": train_result.metrics,
        "dev_metrics": dev_metrics,
        "conditioning_probe": probe,
        "dev_scoring": scoring,
        "decoding": {
            "do_sample": False,
            "num_beams": 1,
            "no_repeat_ngram_size": 0,
            "early_stopping": False,
        },
    }
    (args.output / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "train_log.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
