# Generation Trace-only v9 result record

This directory archives the locally downloaded outputs of the member B
Generation experiment on 2026-11-19. The source run's `run_report.json` was not
available among the downloaded files. The code and frozen data identifiers are
recorded separately; no model weights or environment files are committed.

## Data and method

- Model: offline `google/flan-t5-base` using the existing HPC model folder.
- Frozen data: `project6-6x6-seed42-20261116`; the same 2,000 Generation Train
  records and fixed 100 Generation Dev records used for the v7/v8 comparison.
- `dataset_ids.json` records the ordered Train/Dev sample and environment IDs.
  All 2,000 Train IDs and all 100 Dev IDs are unique, with no overlapping
  Train/Dev environment IDs. No held-out test examples were used here.
- The output target is a single coordinate/action `Trace:` line. The parser
  extracts actions from that line and rejects inconsistent coordinates. It
  does not select candidates, search for a path, or repair predictions.
- Code defaults: seed 42, 3 epochs, batch size 8, gradient accumulation 1,
  constant learning rate 3e-4, one GPU and BF16. The attached training log
  records 750 steps and 3 completed epochs.

SHA-256 of the downloaded transformed data files (not committed here):

```text
Train2000  8b2758f62818575c524031dca6754e2300eb1accaf71abbffddf5e0dbd014e9a
Dev100     3d143c9e07f1d680f7ae343c174f43dbaee4de1cb88670b5a5b7fd62fb047d82
```

## Results

The metric below is a **local diagnostic movement check**, not the official
member A executor result. The saved v7, v8 and v9 predictions contain the
same Dev100 IDs and were replayed with the same local movement rules.

| Method | Dev diagnostic success | Exact match to one reference |
| --- | ---: | ---: |
| v7 grid + state trace, `Actions` output | 48/100 | 47/100 |
| v8 hard-example weighting, `Actions` output | 47/100 | 45/100 |
| v9 Trace-only, parsed actions | **66/100** | **60/100** |

For the existing v7 outputs, parsing the `Trace` instead of `Actions` gives
56/100 local diagnostic successes. Retraining with the Trace-only target gives
66/100. This comparison changes the prompt/output target as well as retraining;
it does not isolate a single causal factor.

The v9 Dev diagnostic success count increased by epoch: 54/100, 60/100 and
66/100. At epoch 3, all 100 outputs parsed and their reported coordinates
agreed with their action tokens. The 34 remaining failures all collided with
obstacles. Long reference paths (length at least 5) succeeded in 20/45 cases,
versus 6/45 for v7 and 10/45 for v8. Required-detour cases succeeded in 1/10,
versus 0/10 for v7 and v8. All 48 v7 successful Dev examples remain successful
in v9, with 18 additional successes. The fixed Train monitor reached 163/200
diagnostic successes. The one-sample save/reload check passed.

## Files

- `epoch_metrics.json`: per-epoch Dev and Train-monitor metrics.
- `dev_predictions.jsonl`: final v9 Dev100 predictions, including raw Trace,
  parsed actions, and the local diagnostic reason.
- `train_monitor_predictions.jsonl`: fixed Train-monitor200 predictions.
- `train_log.json`: training metrics and step history.
- `save_reload_check.json`: one-sample save/reload comparison.
- `dev_predictions_v7_grid_state.jsonl`, `dev_predictions_v8_hard.jsonl`:
  paired Dev100 predictions for the comparison above.
- `run_report_v7_grid_state.json`, `run_report_v8_hard.json`: source run
  metadata for the previous methods.

The next evaluation step is to replay the v9 `plan` values through member A's
official executor and then report ID/OOD held-out results separately. Do not
present the diagnostic 66/100 as an official or held-out test result.
