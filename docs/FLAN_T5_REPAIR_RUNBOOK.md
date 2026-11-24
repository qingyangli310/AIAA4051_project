# Flan-T5 Repair training and ID/OOD evaluation

## Decision rule

`google-t5/t5-small` and `google/flan-t5-*` are different pretrained
checkpoints. Do not rename an existing T5 result as Flan-T5. Select the final
primary family using Development results from a matched baseline comparison,
then fine-tune separate Generation, Verification, and Repair checkpoints from
that same primary base family.

The current Flan-T5 Generator v9 result makes `google/flan-t5-base` a primary
model candidate, not an automatically selected final model. Its archived
immutable revision is:

```text
google/flan-t5-base@7bcac572ce56db69c1ea7c8af255c5d7c9672fc2
```

## Controlled Repair comparison on Development data

The formal Repair v2 data uses 1,000 training rows and 50 Development rows.
The selected run below starts independently from the pinned original
Flan-T5-base snapshot. A matched two-epoch run is retained as selection
evidence; three epochs achieved the better Development Repair Success.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_part1.py \
  --task repair \
  --model flan-t5-base \
  --data-dir data/repair_high_quality_v2 \
  --output results/flan_t5_base_repair_high_quality_v2_epoch3 \
  --epochs 3 \
  --learning-rate 0.0002 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --optimizer adafactor \
  --gradient-checkpointing \
  --load-dtype bfloat16 \
  --not-formal \
  --experiment-label flan_t5_base_repair_high_quality_v2_epoch3
```

On v2, two epochs reached 34/50 successful repairs and three epochs reached
38/50 with lower Development loss. The project limit is at most three passes,
so selecting the three-epoch checkpoint from Development results is
protocol-compliant. The old v1 result remains archived separately and is not
used by the locked v2 pipeline.

For an offline HPC snapshot, add the following argument. The snapshot must
match the model name and revision above and its weight hash must be retained in
the run report.

```text
--model-path /path/to/verified/flan-t5-base
```

Do not use the Generator v9 fine-tuned checkpoint to initialize Repair. Both
task checkpoints should start from the same pretrained Flan-T5-base snapshot
and be fine-tuned independently.

## Required Repair data before the final run

The final frozen Repair training set must:

- stay within the selected 1,000-example allocation;
- use Train environments only;
- keep Dev and Held-out Test environments disjoint;
- contain action substitution, deletion, insertion and premature stopping;
- cover obstacle collision, out of bounds, invalid action token and goal not
  reached;
- include both no-feedback and exact executor-feedback inputs where required;
- use targets that pass the canonical executor;
- record sample IDs, environment IDs and file hashes.

The formal `repair_high_quality_v2` data contains 1,000 Train rows derived from
500 distinct base samples and environments, plus 50 independent Dev rows from
50 distinct base samples and environments. Its base samples have zero overlap
with Generator v9 and Verification high-quality v1. It does not contain the
`invalid_action_token` failure reason and must not be described as complete
four-reason Repair coverage; Verification high-quality v1 does cover all four
required verification reasons.

## Frozen single-error and double-error evaluation

After selecting and locking the primary model, evaluate the same Repair
checkpoint once on both frozen conditions:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_part1_checkpoint.py \
  --task repair \
  --checkpoint /path/to/flan-t5-base-repair/checkpoint-final \
  --data data/part1/repair_id_test.jsonl \
  --output results/flan_t5_base_repair_id

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_part1_checkpoint.py \
  --task repair \
  --checkpoint /path/to/flan-t5-base-repair/checkpoint-final \
  --data data/part1/repair_ood_test.jsonl \
  --output results/flan_t5_base_repair_ood
```

The ID file contains single-error candidates and the OOD file contains paired
double-error candidates. Preserve `evaluation.json` and `predictions.jsonl`
from both commands. Report Success Rate, Feasibility, Optimality and Repair
Success Rate overall and by failure type.

## Current evidence status

| Item | Status | Evidence limit |
|---|---|---|
| Paired ID single-error and OOD double-error data | Complete, 162 pairs | Data construction and executor audit only |
| Symbolic ID/OOD Repair acceptance | Complete | BFS upper bound, not a language-model result |
| Historical BART-base Repair ID/OOD | Reported as 22/162 ID and 5/162 OOD | Summary is committed; per-example evaluation artifacts are not committed |
| T5-small Repair v3 and high-quality v1 | Development only | No frozen ID/OOD result |
| Flan-T5-base Generator v9 | Development only | No official held-out ID result |
| Flan-T5-base Repair v1 | Archived Dev result, 68/90 | Trained on superseded v1 data |
| Flan-T5-base Repair v2 | Dev complete, 38/50 selected at 3 epochs | No frozen ID/OOD result |
| Same-family full Flan-T5 pipeline ID/OOD | Not run | Requires locked Generator, Verifier and Repair checkpoints |

The archived Flan-T5-base v1 two-epoch checkpoint has 68/90 Development Repair
success, 84/90 parsed outputs, 67/90 exact matches and mean optimality 1.0.
The selected v2 three-epoch checkpoint has 38/50 Development Repair success,
47/50 parsed outputs, 33/50 exact matches and mean optimality 1.0. Frozen
ID/OOD and the full same-family cascade remain unrun.

Do not use the symbolic result or the historical BART result as evidence for a
Flan-T5 Repair model. Do not consume the protected test sets until the primary
model, data hashes, decoding and checkpoint hashes are locked from Development
results.
