# Repair v3 deduplicated successor dataset

This directory contains a cleaned successor to the historical Repair v3 training
set. It keeps the same 1,500-row budget and the same 750/750 feedback balance,
but replaces the two repeated model input/target pairs.

- Training rows: 1,500
- Unique `(input_text, target_text)` pairs: 1,500
- Unique candidates: 750
- Train environments: 571
- Output format: Trace only
- Training SHA-256: `1ec7d70869fdaa7bd309889114f1951f7d9e2bb928b55a3e28603c7ba47c7bb5`

The existing Repair v3 checkpoint was trained on the historical dataset with
SHA-256 `c5cc428a0dbc35a6a0ffa2674a0a73134f1669786a081d3b016807bc15404960`.
Do not claim that checkpoint was trained on this cleaned file. Use this directory
for a new training run and record a new checkpoint hash and result report.

Rebuild with:

```bash
python scripts/build_generation_like_repair_data.py \
  --base-train data/processed/nov14/train.jsonl \
  --repair-train data/part1/repair_train.jsonl \
  --repair-dev data/part1/repair_dev.jsonl \
  --output data/repair_generation_like_trace_hardonly_v3_dedup \
  --output-format trace_only \
  --hard-target global_shortest \
  --original-per-feedback 0 \
  --deduplicate-model-io
```
