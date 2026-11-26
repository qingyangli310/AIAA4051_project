# Repair v4 matched baseline on full Dev100

Two separate Flan-T5-base Repair models were trained with seed 42 for three
epochs on 500 matched Train candidates each. The feedback model receives
executor diagnosis and localization; the baseline model receives none. Both
use the same Generator v9 and Verifier checkpoints and the same frozen Dev100
input (SHA-256 `3d143c9e07f1d680f7ae343c174f43dbaee4de1cb88670b5a5b7fd62fb047d82`).
All 100 rows completed with zero runtime errors.

| Condition | End-to-end success | Repairs of 34 Generator failures |
| --- | ---: | ---: |
| Direct | 66/100 | — |
| Independent no-feedback baseline | 68/100 | 2/34 |
| Self feedback | 66/100 | 0/34 |
| Correct executor feedback | 78/100 | 12/34 |
| Shuffled executor feedback | 71/100 | 5/34 |

Correct feedback improves by 10 percentage points over the independent
no-feedback baseline and by 7 points over shuffled feedback. The earlier
70/100 no-feedback result used the feedback-trained checkpoint with feedback
hidden, so this 68/100 matched baseline supersedes that number for model
comparison. Self feedback remains limited by the unchanged Verifier.

The baseline's independent Repair Dev50 result was 18/50 (36%); the feedback
model's was 33/50 (66%). These are diagnostic Dev measurements, not held-out
ID/OOD Test results. See `report.json`, `records.jsonl`, and
`no_feedback_training_report.json` for the machine-readable evidence.
