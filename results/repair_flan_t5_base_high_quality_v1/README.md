# Flan-T5-base high-quality Repair Development result

The selected two-epoch checkpoint repairs 68/90 Development examples. It
outperforms the prior T5-small high-quality run by 4 examples and the matched
Flan-T5-base three-epoch run by 3 examples.

| Model | Epochs | Parse | Exact match | Repair success | Dev loss | Optimality |
|---|---:|---:|---:|---:|---:|---:|
| T5-small | 3 | 80/90 | 62/90 | 64/90 | 0.09066 | 1.0 |
| Flan-T5-base | 3 | 78/90 | 63/90 | 65/90 | 0.07878 | 1.0 |
| Flan-T5-base selected | 2 | 84/90 | 67/90 | 68/90 | 0.04584 | 1.0 |

The two-epoch checkpoint is selected from Development results because the third
epoch increased loss and reduced Repair success. This respects the project cap
of at most three passes over each training subset.

Files:

- `run_report.json`: selected run configuration, hashes and metrics;
- `dev_predictions.jsonl`: all selected-run Dev predictions;
- `train_log.json`: per-step and per-epoch training history;
- `three_epoch_run_report.json`: matched three-epoch comparison.

The run used a memory-efficient configuration because all server GPUs were
shared with long-running services. It used Adafactor, batch size 1, gradient
accumulation 4, BF16 parameter loading and gradient checkpointing. The older
T5-small run used AdamW, so this is not a controlled architecture-only
ablation.

This result is Repair-only and Development-only. It must not be described as a
frozen ID/OOD or end-to-end pipeline result.
