# Repair v4: feedback-localized suffix repair

## Why this replaces the v3 training design

The corrected Dev100 pipeline showed that no-feedback, executor-feedback, and
shuffled-feedback produced identical outputs on all 34 failed Generator
candidates.  Repair v3 also exceeded the Project 6 Repair data limits (1,500
Train and 90 Dev rows) and trained one model on paired prompts with identical
targets, which made ignoring feedback an easy solution.

Repair v4 stays within the stated limits:

- 1,000 materialized Train examples total: 500 no-feedback plus 500 matched
  executor-feedback views;
- 50 unique Dev candidates, evaluated under both views for the focused paired
  ablation;
- at most 3 epochs and seed 42;
- Train environments only for training; no Dev34 or Test examples are used;
- 75% Generation-like obstacle collisions and 25% controlled errors;
- the target preserves the valid prefix and replaces only the failed suffix.

## Build

```bash
python scripts/build_feedback_localized_repair_data.py \
  --base-train data/processed/nov14/train.jsonl \
  --base-dev data/processed/nov14/dev.jsonl \
  --repair-train data/part1/repair_train.jsonl \
  --repair-dev data/part1/repair_dev.jsonl \
  --output data/repair_feedback_localized_v4
```

## Train the two matched models

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_part1.py \
  --task repair \
  --model flan-t5-base \
  --data-dir data/repair_feedback_localized_v4/no_feedback \
  --output results/flan_repair_v4_no_feedback \
  --epochs 3 \
  --experiment-label repair_v4_no_feedback

CUDA_VISIBLE_DEVICES=0 python scripts/train_part1.py \
  --task repair \
  --model flan-t5-base \
  --data-dir data/repair_feedback_localized_v4/executor_feedback \
  --output results/flan_repair_v4_executor_feedback \
  --epochs 3 \
  --experiment-label repair_v4_executor_feedback
```

## Required evaluation

Use `--repair-prompt-protocol feedback_localized_v1` with the new checkpoints.
Compare the two models on the same 50 unique Dev candidates, then run the
executor-feedback checkpoint with correct and shuffled executor feedback.  The
method is accepted only if correct feedback improves conditional Repair success
over both no-feedback and shuffled feedback.  Do not select settings on frozen
ID/OOD Test data.

For a single paired full-pipeline run, pass the feedback model through
`--repair-checkpoint` and the baseline through
`--no-feedback-repair-checkpoint`.  The runner routes only the `no_feedback`
branch to the baseline; self, correct executor, and shuffled executor feedback
all use the feedback-trained model.  Also pass:

```text
--repair-output-format trace_only \
--repair-prompt-protocol feedback_localized_v1
```

After the Dev method is locked, run the selected checkpoint on formal ID
single-error and OOD double-error candidates using the dataset-candidate mode.
