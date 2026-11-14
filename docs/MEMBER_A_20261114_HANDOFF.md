# Member A handoff — 2026-11-14

## Scope and source checks

- Followed Project 6 protocol v1.2 Frozen; seed 42; standard 6x6 only.
- PPNL checkout remained read-only at `5fd0376fc50ae9860bc986085165ebac5675fd44`.
- All seven official files existed, parsed, and matched the Nov 13 SHA-256 audit. The four standard files total 25,080 rows: 25,016 converted and 64 unreachable. The other three files were audited but excluded from main splits; their 447 unreachable rows are still indexed in the excluded manifest. Total excluded: 511, all with unique source file/index pairs.
- No raw PPNL files were copied to or staged from the official data directory.

## Environment identity and split

`environment_id` is `ppnl_` plus SHA-256 of canonical JSON containing only rows, cols and sorted obstacles. Start, goal and plan do not affect it. `base_sample_id` hashes pinned PPNL commit, source filename and source index.

The four standard 6x6 sources yielded 817 unique map layouts. Seed-42 shuffle plus largest remainder produced:

| Split | Environments | Actual ratio | Records |
|---|---:|---:|---:|
| Train | 572 | 70.0122% | 17,510 |
| Dev | 82 | 10.0367% | 2,517 |
| Held-out Test | 163 | 19.9510% | 4,989 |

The environment sets are pairwise disjoint. Held-out records retain `base_sample_id` for the Nov 15 paired conditions.

## Path checks

BFS uses deterministic action order `up`, `down`, `left`, `right`. All 25,016 canonical shortest paths were re-executed and passed `valid=true`, `success=true`; each length equals the BFS distance. All 25,016 official paths also execute successfully; none was longer than the canonical shortest path in these four sources. The deterministic two-step detour produced 25,016 nonoptimal plans, all longer than shortest and Executor-valid/successful.

## Nov 14 shared pilot

`data/samples/nov14_pilot/` contains train, dev, heldout_test, single_error_train, single_error_dev, and a manifest.

| Pilot artifact | Bases / rows |
|---|---:|
| Train base records | 200 |
| Train correct shortest + nonoptimal plans | 200 bases, both plans retained per row |
| Train single-error rows | 800 (200 per corruption type) |
| Dev base records | 20 |
| Dev single-error rows | 80 (20 per corruption type) |
| Held-out base records | 163 available unique environments |

Each error row includes grid, environment/base/source IDs, one corruption type, position, seed, original/corrupted plans and an Executor result. All labels were recomputed with Executor; no corrupted plan succeeded. `valid` remains distinct from `success`; goal-not-reached plans may be valid and unsuccessful. No ID/OOD double-error data was generated.

## Outputs and reproducibility

- Full converted standard splits are in ignored `data/processed/nov14/`.
- Environment manifests are in `data/manifests/{train,dev,heldout_test}.json`.
- All 511 unreachable source rows are in `data/manifests/excluded_manifest.json`.
- The pilot manifest records seed, source commit, splits, ratios, exclusions, path comparisons and corruption counts.
- Re-run with `uv run --python 3.12 python -m src.data.build_nov14_dataset --source ../ppnl_official/ppnl-spatial-temporal-reasoning/single_goal --processed data/processed/nov14 --pilot data/samples/nov14_pilot --seed 42`.

## Tests

`uv run --python 3.12 --with pytest==8.4.2 python -m pytest -q` — **40 passed**.
`git diff --check` — passed. Detailed all-record audit re-executed canonical plans, nonoptimal plans and every pilot corruption; checked split disjointness, counts, budgets and excluded source coverage.

## Deferred work

The 511 excluded source records are recorded but no formal model data is built from them. No Nov 15 paired ID/OOD conditions, double-error generation or model training was run. The 163 held-out pilot bases reflect available unique test environments; the cap is 200.
