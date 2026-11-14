# PPNL source provenance — Member A, 2026-11-13

Upstream: https://github.com/MohamedAghzal/llms-as-path-planners

Pinned commit: `5fd0376fc50ae9860bc986085165ebac5675fd44`.

Audit date: 2026-11-13. Official source was already present locally before this
integration; its original acquisition timestamp was not recorded and cannot be
reliably reconstructed. This integration reads it without changing it. The team
repository was fetched on 2026-11-13 at `83e8927`.

Reproducible acquisition into a **new** directory:

```bash
git clone https://github.com/MohamedAghzal/llms-as-path-planners.git ppnl_source
git -C ppnl_source checkout 5fd0376fc50ae9860bc986085165ebac5675fd44
python scripts/audit_ppnl.py \
  --source ppnl_source/ppnl-spatial-temporal-reasoning/single_goal \
  --output results/ppnl_source_audit_local.json \
  --samples data/samples/ppnl_integration_local.jsonl
```

The raw JSON files must remain outside version control. No original datasets are
copied into this repository. Exact filenames, SHA-256 checksums, counts and all
source-indexed issues are in `results/ppnl_source_audit_20261113.json`.

| File | Records | Missing fields | Conversion halted (unreachable source) |
|---|---:|---:|---:|
| 1_train_set_6x6_samples.json | 16032 | 0 | 43 |
| 1dev_set_6x6_samples.json | 2004 | 0 | 5 |
| 1_goals_test_seen_6x6_samples.json | 2004 | 0 | 3 |
| 1goals_unseen_6x6_samples.json | 5040 | 0 | 13 |
| 1_goals_test_unseen_5x5_samples.json | 3750 | 0 | 61 |
| 1_goals_test_unseen_7x7_samples.json | 3750 | 0 | 0 |
| 1_goals_test_unseen_6x6more_obstacles_samples.json | 4500 | 0 | 386 |

All seven files parsed successfully. Required fields: `world`, `nl_description`,
`solution_coordinates`, `agent_as_a_point`, `agent_has_direction`.
37,080 inputs = 36,569 executable conversions + 511 explicit issues. Every
executable conversion was checked against the Executor and official coordinates.
`Goal not reachable` rows are retained as indexed issues, not converted to empty
or fabricated paths; formal inclusion/exclusion policy needs team confirmation.
The seven integration examples are only converted, traceable samples for API
integration, not a training/dev/test split or extension experiment.

Final November 13 decision (recorded on Nov 14): all 511 unreachable records are
listed with source file/index in `data/manifests/excluded_manifest.json` and are
excluded from normal training and primary-test data. The standard 6x6 files
contain 64 of these rows; the other three audited files contain 447. No raw
official dataset file is copied into this repository.
