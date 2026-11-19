# Repair v3 runbook

## Scope

This run is a post-Part-1 Dev integration experiment. It is explicitly marked
non-formal and must not be reported as held-out evidence.

## Build the fixed-budget data

```bash
python scripts/build_generation_like_repair_data.py \
  --base-train data/processed/nov14/train.jsonl \
  --repair-train data/part1/repair_train.jsonl \
  --repair-dev data/part1/repair_dev.jsonl \
  --output data/repair_generation_like_trace_hardonly_v3 \
  --output-format trace_only \
  --hard-target global_shortest \
  --original-per-feedback 0
```

The builder refuses to overwrite an existing output directory. The committed
manifest records 1,500 rows, 750 unique collision candidates, 571 Train
environments and balanced feedback sources.

## Train

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_part1.py \
  --task repair \
  --model t5-small \
  --data-dir data/repair_generation_like_trace_hardonly_v3 \
  --output results/t5_repair_generation_like_trace_hardonly_v3 \
  --epochs 3 \
  --not-formal \
  --experiment-label generation_like_trace_hardonly_repair_v3
```

## Replay the Generation v9 cascade

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_generation_repair_cascade.py \
  --predictions results/generation_trace_v9/dev_predictions.jsonl \
  --repair-checkpoint \
    results/t5_repair_generation_like_trace_hardonly_v3/checkpoint-final \
  --output results/generation_trace_v9_t5_trace_hardonly_repair_v3_feedback_probe \
  --repair-output-format trace_only \
  --include-shuffled-feedback \
  --batch-size 16 \
  --device cuda:0
```

The evaluator sends only failed Generator candidates to Repair and uses the
canonical Executor before and after Repair. It refuses to overwrite an existing
output directory. The shuffled condition rotates feedback labels by one row and
is diagnostic only.

## Acceptance checks

```bash
python -m pytest -q
git diff --check
```

Before any final project claim, rerun the selected method on frozen held-out
ID/OOD data. Do not describe the current Dev100 result as final test accuracy.

