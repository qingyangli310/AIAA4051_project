# Repair v4 in the full Dev100 pipeline

The same 100 Generator v9 Dev inputs as the corrected v3 run were used (data
SHA-256 `3d143c9e07f1d680f7ae343c174f43dbaee4de1cb88670b5a5b7fd62fb047d82`).
Generator and Verifier checkpoints were unchanged. The Repair checkpoint was
the newly trained Flan-T5-base feedback-localized v4 model. All 100 rows
completed without runtime errors.

| Condition | End-to-end success | Repairs among 34 failed Generator candidates |
| --- | ---: | ---: |
| Direct | 66/100 | — |
| No feedback | 70/100 | 4/34 |
| Self feedback | 66/100 | 0/34 |
| Executor feedback | 78/100 | 12/34 |
| Shuffled executor feedback | 71/100 | 5/34 |

The executor-feedback branch gains 12 percentage points over Direct and 11
points over the old v3 executor-feedback result (67/100). Correct feedback
beats shuffled feedback by 7 points. Shuffled feedback content differed from
the correct step/reason on 32 of the 34 failed candidates. On paired failures,
10 succeeded only with correct feedback and 3 only with shuffled feedback.

The no-feedback branch used the same feedback-trained v4 checkpoint with the
feedback hidden; the separately planned no-feedback v4 baseline has not been
trained. Self feedback still depends on the unchanged Verifier, which routed
too few natural collisions to Repair. These are Dev diagnostic results, not
formal frozen ID/OOD Test results.

The complete machine-readable metrics and per-example outputs are in
`report.json` and `records.jsonl`.
