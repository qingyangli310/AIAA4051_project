# Repair high-quality v1

This 1,000-row Repair dataset replaces the single-error-type hard-only mix with
a validated, diverse, task-aligned mix. It contains 500 semantic candidates,
each represented once without feedback and once with exact executor feedback.

Quality guarantees checked by the builder:

- every candidate actually fails under the canonical executor;
- every executor-feedback label has the exact failure step and reason;
- every target succeeds and has shortest-path length;
- no-feedback prompts contain no diagnosis;
- all complete model input/target pairs are unique;
- Train and Dev environments are disjoint;
- all four corruption types are equally represented;
- failure reasons cover goal-not-reached, obstacle collision, and out-of-bounds.

Training SHA-256:
`15a4853e3e67614e9bcdc00ac83567e88824bee38c61a32721899b1a020a588d`

Rebuild with:

```bash
python scripts/build_high_quality_repair_data.py \
  --source data/samples/member_c_day14/train_repair.jsonl \
  --dev data/part1/repair_dev.jsonl \
  --output data/repair_high_quality_v1 \
  --train-count 1000 \
  --seed 42
```

