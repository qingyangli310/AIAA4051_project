# Integrated Flan-T5-base pipeline Development result

This run combines the selected Flan-T5-base Generator v9, Verification trace
v1, and Repair high-quality v2 checkpoints on the same 100 frozen Generation
Development rows. The Generator runs once per grid and every condition reuses
the same candidate.

| Condition | End-to-end success | Conditional repair success | Feasibility | Mean optimality |
|---|---:|---:|---:|---:|
| Direct Generation | 19/100 (19%) | 0/81 (0%) | 44/100 (44%) | 0.8241 |
| No-feedback Repair | **44/100 (44%)** | **25/81 (30.9%)** | **44/100 (44%)** | 0.9241 |
| Self-feedback Repair | 28/100 (28%) | 10/81 (12.3%) | 40/100 (40%) | 0.9143 |
| Executor-feedback Repair | 43/100 (43%) | 24/81 (29.6%) | 43/100 (43%) | 0.9223 |
| Shuffled executor feedback | **44/100 (44%)** | **25/81 (30.9%)** | **44/100 (44%)** | 0.9241 |

The learned Verifier scored Macro-F1 `0.5074` and first-error/reason accuracy
`0.2469` on the 81 failed Generator candidates routed through self-feedback.

The pipeline integration is technically complete: all 100 rows finished with
zero runtime errors and all five conditions are recorded per sample. The
result does **not** support a claim that feedback content improves Repair:
executor feedback is one success below no feedback, and shuffled feedback ties
no feedback. The self-feedback branch is worse because the learned Verifier is
not accurate enough on Generator-produced failures.

This is Development evidence only. It is not a frozen ID/OOD Test result and
must not be used to tune after formal Test begins.

Artifacts:

- `report.json`: aggregate metrics and checkpoint/data provenance;
- `records.jsonl`: 100 paired per-sample records containing every condition.
