# Repair high-quality v2

This is the formal cross-task-disjoint Repair dataset for the 4,000-example
protocol. It contains 1,000 Train rows and 50 Dev rows.

Train contains 500 semantic candidates from 500 distinct `base_sample_id`
values and 500 distinct environments. Each candidate is represented once with
no feedback and once with exact executor feedback. Those paired rows are a
controlled treatment, not independent samples.

Dev contains 50 rows from 50 distinct base samples and 50 distinct
environments. It therefore contributes exactly 50 independent examples to the
formal 200-example Development budget.

The builder excludes every Generator v9 and Verification high-quality v1 base
sample before selecting Repair data. It also enforces zero Repair Train/Dev
environment overlap, validates every candidate and target with the canonical
executor, and records all source hashes and overlap checks in `manifest.json`.

Rebuild deterministically:

```bash
python scripts/build_high_quality_repair_data.py \
  --exclude-train artifacts/generation_trace_v9/data/transformed_train.jsonl \
  --exclude-train data/verification_high_quality_v1/verification_train.jsonl \
  --exclude-dev artifacts/generation_trace_v9/data/transformed_dev.jsonl \
  --exclude-dev data/verification_high_quality_v1/verification_dev.jsonl \
  --output data/repair_high_quality_v2 \
  --train-count 1000 \
  --dev-count 50 \
  --seed 42
```

Frozen SHA-256 values:

- Train: `a0d826e31a04eea953890d87a7850fa7eafc1dca83777f2b368bbd1414f79069`
- Dev: `f1b9744c3f57ff6fb7c72cf3c42379e628d3f24906d2596f1e22808def8008a0`

The archived v1 Repair checkpoint was trained on different data and must be
retrained before it can be used as the formal v2 Repair checkpoint.
