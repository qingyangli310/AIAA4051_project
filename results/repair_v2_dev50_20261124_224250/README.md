# Repair v2 CPU Development reproduction

The pipeline Repair adapter now uses the Trace instructions shared with the
Repair v2 training-data builder when configured with `output_format="trace_only"`.
Runtime prompts match all 50 frozen Repair Development inputs exactly.

The real CPU inference run reproduced all 50 archived raw model outputs.
Canonical executor success was 38/50, with 47/50 parsed outputs, zero runtime
errors, and 103.39 seconds of inference and scoring. The remaining three
outputs contain inconsistent action/coordinate transitions. The Development
set contains 25 no-feedback and 25 executor-feedback examples.

The selected checkpoint is
`artifacts/repair_flan_t5_base_high_quality_v2/checkpoint-final`, with weight
SHA-256 `c097e2031d30e540e4549c076ce1bf39b475bfa4b4873c518db794c1ad0351d5`.
The original `report.json` and `predictions.jsonl` preserve the local run
configuration, prompts, outputs, parsed paths, and executor results.

From the repository root, download the checkpoint through Git LFS and choose
a fresh output directory:

```powershell
git lfs pull origin --include="artifacts/repair_flan_t5_base_high_quality_v2/checkpoint-final/model.safetensors"
python -u scripts/evaluate_repair_v2_dev.py --checkpoint artifacts/repair_flan_t5_base_high_quality_v2/checkpoint-final --output results/repair_v2_dev50_reproduction --limit 50
```

The sibling `repair_v2_dev1_20261124_221758` directory records the initial
one-example smoke run. These results cover Repair v2's fixed Dev split.
Full-pipeline feedback comparisons and frozen ID/OOD evaluation remain pending.
