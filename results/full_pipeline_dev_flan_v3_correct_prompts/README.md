# Integrated Flan-T5-base pipeline Development result with matched prompts

This run replaces the earlier Development integration result that invoked the
Generator and Repair checkpoints with prompts different from their training
inputs. It uses the archived Flan-T5-base Generator v9, Verification trace v1,
and Repair high-quality v2 checkpoints with their matching Trace prompts on the
same 100 frozen Generation Development rows.

| Condition | End-to-end success | Conditional repair success | Feasibility | Mean optimality |
|---|---:|---:|---:|---:|
| Direct Generation | 66/100 (66%) | 0/34 (0%) | 66/100 (66%) | 0.9933 |
| No-feedback Repair | **67/100 (67%)** | **1/34 (2.9%)** | **67/100 (67%)** | 0.9874 |
| Self-feedback Repair | 66/100 (66%) | 0/34 (0%) | 66/100 (66%) | 0.9933 |
| Executor-feedback Repair | **67/100 (67%)** | **1/34 (2.9%)** | **67/100 (67%)** | 0.9874 |
| Shuffled executor feedback | **67/100 (67%)** | **1/34 (2.9%)** | **67/100 (67%)** | 0.9874 |

All 100 rows completed without runtime errors. Generator output parsed for all
rows and the canonical Executor found 66 successes plus 34 obstacle
collisions. The learned Verifier accepted 33 of the 34 collisions as valid and
routed only one collision to Repair. Across all 100 candidates it achieved
Macro-F1 `0.4241`; first-error accuracy on the 34 collisions was `0/34`, and
error-reason accuracy was `1/34`.

The Repair model generated exactly the same raw output and repaired plan under
no feedback, correct executor feedback, and shuffled executor feedback for all
34 failed candidates. Each condition repaired the same single example. This
run therefore provides no evidence that the Repair checkpoint uses feedback
content. Its independent 38/50 Repair Development result does not transfer to
the Generator's natural collision-only failures.

This is Development evidence only. It is not a frozen ID/OOD Test result.
Formal Test must wait until the dataset-candidate pipeline is locked and the
improvement and ablation choices are frozen from Development evidence.

Artifacts:

- `report.json` contains aggregate metrics, hashes, checkpoint paths, and
  decoding settings.
- `records.jsonl` contains all five paired conditions for every Dev row.
