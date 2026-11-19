# Repair v3 Dev integration result

Repair v3 is a T5-small trace-only model trained on 1,500 Train-only,
Generator-like obstacle-collision examples. The data budget is unchanged from
the Part 1 Repair baseline. Each of 750 collision candidates appears once with
no feedback and once with Executor feedback.

## Dev100 cascade

| Condition | Repair success on 34 Generator failures | End-to-end success |
|---|---:|---:|
| Generator v9 without Repair | N/A | 66/100 |
| Repair v3, no feedback | 12/34 | 78/100 |
| Repair v3, Executor feedback | 14/34 | 80/100 |
| Repair v3, shuffled Executor feedback | 14/34 | 80/100 |

The official deterministic Executor agrees with the archived Generator v9
local diagnostic on all 100 examples. Train and Dev environment overlap is
zero.

After adding Trace-aware scoring, the frozen Repair Dev90 checkpoint evaluation
reports a 71/90 parse count and 54/90 successful repairs. The original training
report is retained as provenance, but its legacy plan-only scorer incorrectly
shows zero parsed Trace outputs; use `dev_rescored_report.json` for the corrected
checkpoint score.

This is a Dev diagnostic, not held-out test evidence. Correct and shuffled
feedback produce the same output on 33/34 failed candidates, so the result does
not establish that the model uses the actual `first_error_step` value. The
supported conclusion is that Repair raises this Dev cascade from 66% to
78–80%, while feedback-content dependence remains unproven.

Large checkpoints and per-example raw logs are intentionally excluded from
Git. The selected checkpoint remains on the project CUDA host; its training
metadata and data hashes are recorded in this directory.

