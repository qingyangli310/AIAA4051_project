# Member C handoff: 2026-11-13 to 2026-11-14

## Completed on November 13

- Repair request and output validation use the frozen schema in `src/repair_schema.py`.
- `no_feedback`, `self_feedback`, and `executor_feedback` share one interface.
- `scripts/run_c_day13_mock.py` produces five complete
  Generate-Verify-Repair-Execute logs. Every row is explicitly marked `mock`.
- The result CSV schema remains `results/result_template.csv`.
- Pipeline failures remain isolated per sample through `run_pipeline_batch`.

Run the evidence:

```bash
python scripts/run_c_day13_mock.py
python -m pytest -q tests/test_pipeline_smoke.py
```

## Completed on November 14

- `src/repair/dataset.py` converts A's Repair task rows into two supervised
  conditions by default: `no_feedback` and `executor_feedback`.
- `self_feedback` uses the same schema but is generated only when B supplies a
  real self-verifier result. It is never fabricated from executor feedback.
- The model input contains the grid, corrupted candidate and permitted feedback;
  the correct repaired path appears only in the supervised output.
- `src/evaluation/metrics.py` implements Success, Feasibility, Optimality,
  Verification Macro-F1, first-error accuracy, Repair Success Rate and
  End-to-End Success Rate without adding third-party dependencies.
- Empty eligible sets return `None`, preventing a misleading zero from being
  reported as a measured score.
- The 6x6 November 14 conversion produced 1,600 Train and 160 Dev Repair rows;
  the inspectable summary is `results/member_c_day14_data_stats.json`.

Build the small Train/Dev Repair data:

```bash
python scripts/build_repair_data.py \
  data/samples/nov14_pilot/single_error_train.jsonl \
  data/samples/member_c_day14/train_repair.jsonl
python scripts/build_repair_data.py \
  data/samples/nov14_pilot/single_error_dev.jsonl \
  data/samples/member_c_day14/dev_repair.jsonl
python -m pytest -q tests/test_repair_dataset.py tests/test_evaluation_metrics.py
```

## Interface notes for A and B

- A: Repair construction accepts `grid_size` or `rows`/`cols`, but emits only
  the frozen `rows`/`cols` schema. A's executor `valid=True, success=False` for
  `goal_not_reached` is normalized to Verification-task `valid=False`.
- A: the November 14 6x6 pilot contains stable `base_sample_id`,
  `source_plan_id`, and `corruption_seed`; all are preserved in C's output.
- B: provide `self_feedback` as `valid`, `first_error_step`, and `reason`.
  Parser failures must remain explicit and must not be replaced with executor
  feedback.
- B: `first_error_step` is zero-based and `goal_not_reached` must use
  `len(candidate_plan)`.

## Remaining external input

B's real `self_feedback` is not a November 14 blocker. The schema is ready and
will generate that condition once B supplies actual verifier predictions for
the November 15 integration run.
