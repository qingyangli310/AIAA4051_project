# Member A handoff — 2026-11-16 data freeze

## Frozen release

- Freeze ID: `project6-6x6-seed42-20261116`
- Protocol: v1.2 Frozen; seed 42; standard 6×6 main data only
- PPNL repository: `https://github.com/MohamedAghzal/llms-as-path-planners.git`
- PPNL commit: `5fd0376fc50ae9860bc986085165ebac5675fd44`
- Data generation code version: `c2a223f310fb053e1583dc26c069f2a429445427`
- Freeze manifest: `data/manifests/frozen_dataset_20261116.json`
- Audit result: `results/data_freeze_audit_20261116.json`

The existing Nov 13–15 data was audited in place. No dataset generator was run to rebuild or overwrite those accepted datasets. The complete converted files in ignored `data/processed/nov14/` were present and hashed. The PPNL checkout remained read only; no raw source JSON is tracked in Git.

## PPNL source audit

All seven official files parsed, had all five required fields, matched the expected record count and SHA-256, and had only the documented unreachable-source conversion exclusions.

| File | Records | Converted | Unreachable | SHA-256 |
|---|---:|---:|---:|---|
| `1_train_set_6x6_samples.json` | 16,032 | 15,989 | 43 | `b20a44269cff617abd5132d3ad209f72fb0ce92f78b4eaefc5aa8a92fac093e2` |
| `1dev_set_6x6_samples.json` | 2,004 | 1,999 | 5 | `aad6f181c4eff903c1f262ae009f2c3ac2d0f787abae4765c51976fd1ee8c574` |
| `1_goals_test_seen_6x6_samples.json` | 2,004 | 2,001 | 3 | `ebdc305d207b0fda3d147b76ddd93fdecdb879dd5bd6acfad90bd242c15a2507` |
| `1goals_unseen_6x6_samples.json` | 5,040 | 5,027 | 13 | `ee841e3adb907116471989331f0e4cffed16038c46592491abdd7bafd34d432e` |
| `1_goals_test_unseen_5x5_samples.json` | 3,750 | 3,689 | 61 | `f34edfe112d248087ae76366c4f4e56cc0e0d31729d0bd981b6c8656344b32cd` |
| `1_goals_test_unseen_7x7_samples.json` | 3,750 | 3,750 | 0 | `db358fd01da87004c2d62551e44cfcb506cb00e88aba29815ebc0fd1c288eb58` |
| `1_goals_test_unseen_6x6more_obstacles_samples.json` | 4,500 | 4,114 | 386 | `8825fee0599751b4da946b9b6a75808bd9aeff8a5595e8db9e833ee38828b16d` |

Totals: **37,080** source records, **36,569** convertible records, and **511** unreachable records. For the four standard 6×6 files: **25,080** records, **25,016** convertible, and **64** unreachable. All 511 exclusions have a unique `source_file`/`source_index` pair in `data/manifests/excluded_manifest.json`.

## Split, paired tests, and path audit

| Split | Environments | Records | Actual environment ratio |
|---|---:|---:|---:|
| Train | 572 | 17,510 | 70.0122% |
| Dev | 82 | 2,517 | 10.0367% |
| Held-out Test | 163 | 4,989 | 19.9510% |

The Train, Dev, and Held-out environment sets are pairwise disjoint. The complete local processed data was available; all **25,016** shortest paths and **25,016** nonoptimal paths were re-executed. Every shortest path was valid and successful with the stored shortest length; every nonoptimal path was valid, successful, and strictly longer. Both paths for all **163** Nov 14 Held-out pilot bases also passed Executor.

The paired Nov 15 test condition remains **162 ID + 162 OOD + 162 pairs**, with one documented exception. The four ID types are substitution 41, deletion 41, insertion 40, and premature stopping 40. OOD combination counts are in the freeze manifest and the Nov 15 manifest. Full audit re-executed all **324** final ID/OOD labels, replayed **324** independent OOD mutations, replayed **162** two-operation provenance sequences, and confirmed all 162 combined OOD plans execute fully with both edits observable. ID/OOD base IDs and environment sets match; neither test condition overlaps Train or Dev.

The freeze manifest stores SHA-256 values for all derived split JSONL files, split manifests, excluded manifest, Nov 14 pilot artifacts, and Nov 15 paired artifacts. The three full converted JSONL split files remain ignored by Git and are not included in this commit.

## Executor semantics and C review

The real A Executor intentionally keeps these meanings:

- `valid`: no illegal action or movement-rule violation;
- `success`: the plan reaches the goal;
- `goal_not_reached`: `valid=true`, `success=false`, and `first_error_step=len(plan)`.

C's current Mock Verifier returns `valid=false` for `goal_not_reached`, as documented in its existing interface report. The meanings are not conflated: the new `tests/test_executor_compatibility.py` pins both outputs, and README documents that downstream code should inspect `success` and `reason`. No C-owned module was modified.

Reviewed C branches and written material:

- `feature/c-repair-pipeline` at `e3927b64155c4062e2d53dbcc101c9b7a58cf5c9`; C's report is `docs/成员C_11月10日至12日工作说明.md`.
- `feature/c-repair-evaluation` at `608525e11ad2c25db42b35a4e826b829765ee995`.
- The C report describes the Mock prototype and lists suggested future review points; it contains no formal A-specific review findings. No additional written C review was found locally or in GitHub PR/Issue records at audit time.

The existing semantic difference is a cross-member interface item for C to resolve when replacing the Mock Verifier with the real Executor. This freeze preserves the agreed Executor meaning and does not change C's code.

## Commands and results

```bash
UV_CACHE_DIR=/tmp/uv-cache-p6 uv run --offline --python 3.12 scripts/audit_frozen_data.py
UV_CACHE_DIR=/tmp/uv-cache-p6 uv run --offline --python 3.12 scripts/demo_executor_and_leakage.py
UV_CACHE_DIR=/tmp/uv-cache-p6 uv run --offline --python 3.12 --with pytest==8.4.2 python -m pytest -q
git diff --check
```

Results: frozen data audit passed; both requested demos ran through `src.executor.executor.execute_plan`; the correct path reached its goal; the error-path example reported `goal_not_reached`, step 4, `valid=true`, `success=false`; Train/Dev/Held-out intersections were all zero; ID/OOD pairing was 162/162/162. Full tests: **49 passed**. `git diff --check` passed.

## Remaining cross-member item

C's Mock Verifier currently uses `valid=false` to route goal-not-reached candidates into Repair, while A's Executor uses `valid=true, success=false` to represent feasible but incomplete plans. The compatibility test and documentation preserve this distinction. The C owner should decide the final adapter/control-flow mapping when integrating the real Executor; no C code was changed as part of this freeze.
