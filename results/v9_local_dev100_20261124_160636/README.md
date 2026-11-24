# Local v9 Generator Dev100 reproduction

This CPU run evaluated the fixed v9 Development set through the pipeline's
Generator adapter and canonical executor after connecting the adapter to the
original v9 Trace prompt. All 100 raw model outputs and parsed action lists
matched the archived v9 predictions.

- Canonical executor success was 66/100.
- All 100 outputs parsed successfully, with zero runtime errors.
- Exact match to the reference action list was 60/100.
- The 34 unsuccessful paths collided with obstacles.
- Inference and scoring took 627.71 seconds on CPU.
- Python 3.12.6, PyTorch 2.6.0, Transformers 4.57.6, and greedy
  decoding were used, with one beam and a 128-token output limit.

The selected Generator checkpoint is the existing
`artifacts/generation_trace_v9/checkpoint-final` archive. Download its weights
with Git LFS before running the model. The local absolute checkpoint path in
`report.json` records the directory used for this run.

To reproduce from the repository root, choose a new output directory:

```powershell
python -u scripts/evaluate_v9_dev.py --checkpoint artifacts/generation_trace_v9/checkpoint-final --output results/v9_local_dev100_reproduction
```

`predictions.jsonl` preserves prompts, model outputs, parsed plans, executor
results, and per-example archive comparisons. `report.json` contains the
aggregate counts, data hash, and decoding configuration. These results cover
the fixed Generation Dev100 set. Held-out evaluation and the complete
Generate Verify Repair comparison remain pending.
