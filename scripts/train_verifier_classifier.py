"""Train a structured T5-encoder verifier for at most three epochs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.part1.data import read_jsonl


MODEL_NAME = "google-t5/t5-small"
MODEL_REVISION = "df1b051c49625cf57a3d0d8d3863ed4d13564fe4"
REASONS = (
    "goal_not_reached",
    "obstacle_collision",
    "out_of_bounds",
    "invalid_action_token",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def macro_f1(actual: list[int], predicted: list[int]) -> float:
    values = []
    for label in (0, 1):
        true_positive = sum(a == label and p == label for a, p in zip(actual, predicted))
        false_positive = sum(a != label and p == label for a, p in zip(actual, predicted))
        false_negative = sum(a == label and p != label for a, p in zip(actual, predicted))
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(values) / len(values)


def compose_prediction(
    is_valid: bool, reason_index: int, step_index: int, plan_length: int
) -> str:
    if is_valid:
        return "valid"
    reason = REASONS[reason_index]
    if reason == "goal_not_reached":
        step = plan_length
    else:
        step = min(step_index, max(plan_length - 1, 0))
    return f"invalid | step={step} | reason={reason}"


def score(rows: list[dict[str, Any]], outputs: list[str]) -> dict[str, Any]:
    actual_valid = [int(row["executor_label"]["success"]) for row in rows]
    predicted_valid = [int(output == "valid") for output in outputs]
    invalid_indices = [index for index, value in enumerate(actual_valid) if not value]
    reason_correct = sum(
        f"reason={rows[index]['executor_label']['reason']}" in outputs[index]
        for index in invalid_indices
    )
    step_correct = sum(
        f"step={rows[index]['executor_label']['first_error_step']}" in outputs[index]
        for index in invalid_indices
    )
    exact = sum(row["target_text"] == output for row, output in zip(rows, outputs))
    return {
        "examples": len(rows),
        "valid_invalid_macro_f1": macro_f1(actual_valid, predicted_valid),
        "valid_invalid_accuracy": sum(a == p for a, p in zip(actual_valid, predicted_valid)) / len(rows),
        "first_error_accuracy": step_correct / len(invalid_indices),
        "error_reason_accuracy": reason_correct / len(invalid_indices),
        "exact_match_rate": exact / len(rows),
        "prediction_counts": dict(Counter(outputs)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data/verification_high_quality_v1"
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--experiment-label", default="verification_high_quality_v1_classifier")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 3:
        raise ValueError("formal Verification training is limited to one to three epochs")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoTokenizer, T5EncoderModel
    except ImportError as exc:
        raise RuntimeError("install requirements-project6.txt in the GPU environment") from exc
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("formal Verification training requires CUDA")

    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    train_path = args.data_dir / "verification_train.jsonl"
    dev_path = args.data_dir / "verification_dev.jsonl"
    train_rows = read_jsonl(train_path)
    dev_rows = read_jsonl(dev_path)
    max_step = max(
        len(row["candidate_plan"]) for row in train_rows + dev_rows
    )
    model_source = str(args.model_path) if args.model_path else MODEL_NAME
    revision = None if args.model_path else MODEL_REVISION
    tokenizer = AutoTokenizer.from_pretrained(model_source, revision=revision)

    class VerificationDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]):
            self.rows = rows
            self.encoded = []
            for row in rows:
                encoded = tokenizer(
                    row["input_text"], truncation=False, max_length=args.max_input_length
                )
                if len(encoded["input_ids"]) > args.max_input_length:
                    raise ValueError("Verification input exceeds max_input_length")
                label = row["executor_label"]
                self.encoded.append(
                    {
                        "input_ids": encoded["input_ids"],
                        "attention_mask": encoded["attention_mask"],
                        "valid_label": int(label["success"]),
                        "reason_label": (
                            REASONS.index(label["reason"]) if not label["success"] else -100
                        ),
                        "step_label": (
                            int(label["first_error_step"]) if not label["success"] else -100
                        ),
                    }
                )

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index: int):
            return self.encoded[index]

    def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
        padded = tokenizer.pad(
            [
                {
                    "input_ids": item["input_ids"],
                    "attention_mask": item["attention_mask"],
                }
                for item in items
            ],
            padding=True,
            return_tensors="pt",
        )
        for field in ("valid_label", "reason_label", "step_label"):
            padded[field] = torch.tensor([item[field] for item in items], dtype=torch.long)
        return padded

    class StructuredVerifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = T5EncoderModel.from_pretrained(model_source, revision=revision)
            hidden = self.encoder.config.d_model
            self.dropout = nn.Dropout(0.1)
            self.valid_head = nn.Linear(hidden, 2)
            self.reason_head = nn.Linear(hidden, len(REASONS))
            self.step_head = nn.Linear(hidden, max_step + 1)

        def forward(self, input_ids, attention_mask):
            hidden = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
            pooled = self.dropout(pooled)
            return (
                self.valid_head(pooled),
                self.reason_head(pooled),
                self.step_head(pooled),
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StructuredVerifier().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        VerificationDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        VerificationDataset(dev_rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())

    def predict(loader: DataLoader, rows: list[dict[str, Any]]) -> tuple[list[str], float]:
        model.eval()
        outputs: list[str] = []
        losses = []
        with torch.inference_mode():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                valid_label = batch["valid_label"].to(device)
                reason_label = batch["reason_label"].to(device)
                step_label = batch["step_label"].to(device)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
                ):
                    valid_logits, reason_logits, step_logits = model(
                        input_ids, attention_mask
                    )
                    loss = F.cross_entropy(valid_logits, valid_label)
                    invalid = reason_label != -100
                    if invalid.any():
                        loss = loss + F.cross_entropy(
                            reason_logits[invalid], reason_label[invalid]
                        ) + F.cross_entropy(step_logits[invalid], step_label[invalid])
                losses.append(loss.item())
                valid_predictions = valid_logits.argmax(-1).tolist()
                reason_predictions = reason_logits.argmax(-1).tolist()
                step_predictions = step_logits.argmax(-1).tolist()
                start = len(outputs)
                for offset, (valid_value, reason_value, step_value) in enumerate(
                    zip(valid_predictions, reason_predictions, step_predictions)
                ):
                    plan_length = len(rows[start + offset]["candidate_plan"])
                    outputs.append(
                        compose_prediction(
                            bool(valid_value), reason_value, step_value, plan_length
                        )
                    )
        return outputs, sum(losses) / len(losses)

    args.output.mkdir(parents=True)
    history = []
    best_key = None
    best_state = None
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            valid_label = batch["valid_label"].to(device)
            reason_label = batch["reason_label"].to(device)
            step_label = batch["step_label"].to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                valid_logits, reason_logits, step_logits = model(
                    input_ids, attention_mask
                )
                loss = F.cross_entropy(valid_logits, valid_label)
                invalid = reason_label != -100
                loss = loss + F.cross_entropy(
                    reason_logits[invalid], reason_label[invalid]
                ) + F.cross_entropy(step_logits[invalid], step_label[invalid])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        dev_outputs, dev_loss = predict(dev_loader, dev_rows)
        dev_scoring = score(dev_rows, dev_outputs)
        key = (
            dev_scoring["valid_invalid_macro_f1"],
            dev_scoring["first_error_accuracy"],
            dev_scoring["error_reason_accuracy"],
            -dev_loss,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": sum(epoch_losses) / len(epoch_losses),
                "dev_loss": dev_loss,
                "dev_scoring": dev_scoring,
            }
        )
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    duration = time.perf_counter() - started
    assert best_state is not None
    model.load_state_dict(best_state)
    final_outputs, final_dev_loss = predict(dev_loader, dev_rows)
    final_scoring = score(dev_rows, final_outputs)
    best_epoch = max(
        history,
        key=lambda row: (
            row["dev_scoring"]["valid_invalid_macro_f1"],
            row["dev_scoring"]["first_error_accuracy"],
            row["dev_scoring"]["error_reason_accuracy"],
            -row["dev_loss"],
        ),
    )["epoch"]

    checkpoint = args.output / "checkpoint-final"
    checkpoint.mkdir()
    model.encoder.save_pretrained(checkpoint / "encoder")
    tokenizer.save_pretrained(checkpoint / "tokenizer")
    torch.save(
        {
            "valid_head": model.valid_head.state_dict(),
            "reason_head": model.reason_head.state_dict(),
            "step_head": model.step_head.state_dict(),
            "reasons": REASONS,
            "max_step": max_step,
        },
        checkpoint / "classifier_heads.pt",
    )
    prediction_rows = [
        {
            "example_id": row["example_id"],
            "environment_id": row["environment_id"],
            "target_text": row["target_text"],
            "raw_output": output,
        }
        for row, output in zip(dev_rows, final_outputs)
    ]
    (args.output / "dev_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    report = {
        "formal_part1_run": True,
        "method": "structured_t5_encoder_multitask_classifier",
        "experiment_label": args.experiment_label,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "model_load_source": model_source,
        "seed": 42,
        "device": str(device),
        "bf16_autocast": use_bf16,
        "train_count": len(train_rows),
        "dev_count": len(dev_rows),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "optimizer": "adamw_torch",
        "train_seconds": duration,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "train_sha256": sha256_file(train_path),
        "dev_sha256": sha256_file(dev_path),
        "history": history,
        "dev_loss": final_dev_loss,
        "dev_scoring": final_scoring,
        "output_format": "valid or invalid | step=<integer> | reason=<error_type>",
    }
    (args.output / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
