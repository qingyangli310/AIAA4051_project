# Member A handoff — 2026-11-15

## Scope

Generated paired ID single-error and OOD double-error conditions from the 163 unique held-out bases in `data/samples/nov14_pilot/heldout_test.jsonl`. The Nov 14 source data was read only. The generator uses seed 42 and does not train models or change other members' modules.

## Results

| Artifact | Rows |
|---|---:|
| ID single-error | 162 |
| OOD double-error | 162 |
| Paired base samples | 162 |
| Unpairable exceptions | 1 |

All 162 retained records have one-to-one matching `base_sample_id` and `environment_id`. Their environment IDs are disjoint from Train (572 environments) and Dev (82 environments). The single exception is recorded in `exceptions.jsonl`; no base was duplicated to meet a target count. It could not meet the strict double-corruption criteria after candidate search across the supported mutation combinations.

ID corruption distribution is balanced to within one sample:

| Corruption type | Count |
|---|---:|
| `action_substitution` | 41 |
| `action_deletion` | 41 |
| `action_insertion` | 40 |
| `premature_stopping` | 40 |

OOD tries different-type pairs first and uses same-type pairs when needed. Ordered combination counts: deletion+insertion 13, deletion+substitution 10, insertion+deletion 25, insertion+insertion 1, insertion+substitution 57, substitution+deletion 4, substitution+insertion 21, premature-stopping+deletion 5, premature-stopping+insertion 16, and premature-stopping+substitution 10. Each operation targets a distinct original logical position. For each mutation, provenance stores its operation ID, type, logical position, index, deterministic seed, before/after plan, independent single-mutation plan, and Executor result for that independent plan. Each individual result is a valid but unsuccessful `goal_not_reached`. Sequential replay matches the final corrupted plan exactly. The combined Executor result is valid, unsuccessful, and executes every action. For each operation, audit removes that operation, replays the other one alone, and confirms the combined trace/final state differs, demonstrating an observable execution effect.

## Files

- Generator: `src/data/build_nov15_paired_conditions.py`
- Pilot: `data/samples/nov15_paired/{id_single_error.jsonl,ood_double_error.jsonl,pair_manifest.jsonl,manifest.json,exceptions.jsonl}`
- Protocol manifests: `data/manifests/{id_single_error,ood_double_error}.jsonl`
- Tests: `tests/test_paired_conditions.py`, `tests/test_double_corruption_provenance.py`, `tests/test_id_ood_pairing.py`

The data rows include grid, shortest and nonoptimal reference plans, source-plan ID, corruption metadata, corrupted plan, and Executor label. OOD provenance includes full traces and labels for each independent operation and the combined plan.

## Audit and verification

- Full audit independently re-executed all 162 ID and 162 OOD plans: **324 labels matched** the Executor.
- OOD independent operation replays: **324**, all failed as valid `goal_not_reached` plans.
- OOD sequential provenance replay: **162/162** exactly matched final corrupted plans.
- Distinct logical positions and full combined execution: **162/162**; no second mutation was hidden by early termination.
- Pair count, IDs, environment sets, Train/Dev isolation, and Manifest counts: passed.
- Fixed-seed reproducibility: the test generated the outputs twice in isolated temporary directories and compared all five artifact byte streams; identical.
- Tests: `UV_CACHE_DIR=/tmp/uv-cache-p6 uv run --offline --python 3.12 --with pytest==8.4.2 python -m pytest -q` — **46 passed**.
- `git diff --check` — passed.

The requested online pytest invocation could not resolve PyPI DNS in this environment. The same Python 3.12 and pinned pytest 8.4.2 test suite passed using the already cached package in offline mode.

## Reproduction

From the repository root:

```bash
uv run --python 3.12 python -m src.data.build_nov15_paired_conditions \
  --input data/samples/nov14_pilot/heldout_test.jsonl \
  --output data/samples/nov15_paired \
  --manifests data/manifests \
  --seed 42
```

The builder requires exactly 163 unique held-out bases, verifies both reference paths with Executor, rejects Train/Dev environment overlap, and writes the ID/OOD outputs only after candidate construction. The output cap is 200 per condition; the generated 162 rows per condition are below the cap.

## Remaining note

OOD pairs cover multiple different-type combinations, with same-type combinations retained where needed. One excluded base is retained in the exception file with its reason. No Nov 16 freeze work was performed.
