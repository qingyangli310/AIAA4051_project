# Generation Trace-only v9

This package isolates the copy-error question from v7. It reuses the frozen
Train2000, fixed Dev100, Flan-T5-base assets, seed 42, batch size 8, constant
learning rate 3e-4 and 3 epochs. No additional examples are created.

## 1. Re-score the existing v7 checkpoint (no GPU)

Upload the whole `generation_trace_v9` directory under `day14_hpc_v1`.
From the `day14_hpc_v1` shell, identify the existing v7 run:

```bash
ls -td outputs/state_v7_grid_state_*
```

Replace `V7_JOBID` with the correct number:

```bash
python generation_trace_v9/evaluate_existing_trace.py \
  --predictions outputs/state_v7_grid_state_V7_JOBID/dev_predictions.jsonl \
  --output outputs/trace_v9_existing_V7_JOBID
cat outputs/trace_v9_existing_V7_JOBID/trace_comparison_report.json
```

This reads saved predictions and compares the original `Actions` line with
actions extracted from `Trace` on exactly the same examples. The output is
diagnostic and does not claim official A executor success.

## 2. Train one-output Trace model (GPU)

The existing `generation_models_v5/model_assets.py`, frozen Train2000,
Dev100 and Flan-T5-base files must be in their prior locations. Submit:

```bash
sbatch generation_trace_v9/run_trace.sbatch
```

Then, using the job ID printed by `sbatch`:

```bash
squeue -u "$USER"
cat logs/gen_trace_JOBID.err
ls outputs/trace_v9_trace_only_JOBID
cat outputs/trace_v9_trace_only_JOBID/run_report.json
```

The job writes `epoch_metrics.json`, per-epoch checkpoints and predictions,
`checkpoint-final`, `dev_predictions.jsonl`, `train_monitor_predictions.jsonl`,
`save_reload_check.json` and `run_report.json`. The final `plan` field contains
actions parsed from Trace. The report also includes a local diagnostic movement
success rate. Export these predictions to the team's official A executor for
final success rates. Do not use diagnostic success as the official result. The
trainer does not use an executor to select or repair paths.
