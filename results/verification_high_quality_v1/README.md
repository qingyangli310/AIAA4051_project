# High-quality Verification v1 results

## Selected run

The selected model is Flan-T5-base trained with supervised execution traces and
a final strict verdict. It completed exactly three epochs on CUDA.

| Dev metric | Result |
|---|---:|
| Parse rate | 100% |
| Strict verdict exact match | 70% |
| Verification Macro-F1 | 0.8572 |
| First-error-step accuracy | 44% |
| Error-reason accuracy | 52% |
| Eval loss | 0.0333 |

These values are Dev-only. The frozen ID/OOD tests were not used.

The raw training-time report recorded zero parse rate because the installed
parser expected `Verdict:` to start on a new line while T5 normalized the line
break to a space. `run_report.json` rescored the unchanged saved predictions
after the parser was made compatible with both whitespace forms. The raw report
is retained as `run_report.raw.json`; no prediction was regenerated.

## Compared three-epoch runs

| Representation / model | Dev Macro-F1 | First-error | Reason |
|---|---:|---:|---:|
| Direct verdict, Flan-T5-base + Adafactor | 0.5003 | 4% | 16% |
| Direct verdict, T5-small + AdamW | 0.5559 | 8% | 24% |
| State tracking input, T5-small + AdamW | 0.5559 | 24% | 24% |
| Trace then verdict, Flan-T5-base + AdamW | **0.8572** | **44%** | **52%** |

## Dataset quality controls

- Exactly 1,000 training examples and 50 Dev examples.
- Train is balanced 500 valid / 500 invalid.
- Invalid training examples are balanced at 125 per required error reason.
- All 1,000 training `base_sample_id` values and model input/output pairs are unique.
- Zero `base_sample_id` or `(environment_id, candidate_plan)` overlap with the
  frozen Generator v9 and Repair high-quality v1 training selections.
- Zero environment overlap between Verification Train and Verification Dev.
- Every stored verdict was recomputed with the deterministic canonical executor.

Generator v9 already spans all 572 available training environments, so making
Verification use entirely different environments from Generator is impossible.
The stricter feasible leakage controls above prevent reuse of examples and plans.
