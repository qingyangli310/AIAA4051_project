# Repair Dev50 through the shared evaluation entry point

This real CPU run validates `scripts/evaluate_part1_checkpoint.py` with
`--task repair --repair-output-format trace_only`. The evaluation entry point
uses the pipeline Repair adapter to construct the training-compatible prompt
from the grid, frozen candidate, and supplied feedback. Reference paths are
used only for scoring after inference.

- Success and Repair Success were 38/50 (76%).
- Parse success was 47/50 (94%), and feasibility was 39/50 (78%).
- Exact reference Trace match was 33/50 (66%).
- All 50 prompts and raw outputs matched the archived Repair v2 Dev records.
- Mean optimality across the 38 successful paths was 1.0.
- Inference took 83.76 seconds on CPU.

`evaluation.json` contains the original report and data hash.
`predictions.jsonl` preserves the actual prompts, raw outputs, parsed paths,
and parsing status. The generic `end_to_end_success_rate` field in this
Repair-only report is not a measured Generate Verify Repair pipeline result.

From the repository root, use a fresh output directory:

```powershell
python -u scripts/evaluate_part1_checkpoint.py --task repair --repair-output-format trace_only --checkpoint artifacts/repair_flan_t5_base_high_quality_v2/checkpoint-final --data data/repair_high_quality_v2/repair_dev.jsonl --output results/repair_unified_dev50_reproduction --allow-cpu
```

The frozen ID/OOD commands in `docs/FLAN_T5_REPAIR_RUNBOOK.md` now include the
same Trace output flag. This run validates the fixed Dev50 evaluation path.
Formal ID/OOD evaluation remains pending.
