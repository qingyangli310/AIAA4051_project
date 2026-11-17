# Member A training data handoff — 2026-11-17

## Frozen source data

Freeze: `project6-6x6-seed42-20261116` (seed 42). The shared Train and Dev files match the immutable hashes in `data/manifests/frozen_dataset_20261116.json`.

| Split | Rows | Environments | SHA-256 |
|---|---:|---:|---|
| Train | 17,510 | 572 | `765e66a975faffedc28893f23eb50dd5b7d1c68e4ee5f27ec08dcb3d842cdd4c` |
| Dev | 2,517 | 82 | `7c3f490b65e21b005fdde4b2d70c1d74dd1db655b065d505c5ce1be28db2f20f` |

Each JSONL row parsed. Required fields include `environment_id`, map dimensions and coordinates, obstacles, source file/index, `base_sample_id`, official and shortest plan data, Executor label, and nonoptimal path data. Every `environment_id` was recomputed from rows/cols/sorted obstacles; every `base_sample_id` was recomputed from the frozen PPNL commit, source filename and index. IDs are unique within each split, and source file/index pairs are unique within each split. All rows map to the corresponding split manifest. Train and Dev have zero environment overlap. They also do not overlap the Held-out environment manifest.

The shared Train file contains 17,510 rows. No training-budget subsample was selected or modified by this handoff; the existing project task cap is 4,000 training examples, so B/C should choose and document any downstream sampling under the agreed budget before formal training.

## GitHub access status

Remote branch trees were inspected read-only after fetching refs; no B/C branch was checked out or modified.

| Branch | Head observed | Train + Dev present | Held-out present |
|---|---|---|---|
| `feature/a-executor` | `e7fa80e631644482f82c23618c3cd5f6124ae97a` | Yes | No |
| `feature/c-repair-evaluation` | `6e3c5e999ba0799411b04c3ce6fcfbd43096842c` | Yes | No |
| `feature/c-repair-pipeline` | `e3927b64155c4062e2d53dbcc101c9b7a58cf5c9` | No | No |
| `feature/b-verifier` | `64c49dac3a5b1cc2fe05dd46d8fd8bc98db61e37` | No | No |

Thus the current C evaluation branch has the files, while C's pipeline branch and B's verifier branch do not. Held-out was not found on any inspected branch. This report does not sync branches; B/C should merge or cherry-pick the A data-sharing commit if they need direct files in their branch.

## Reproducible audit

The read-only auditor is `scripts/audit_shared_training_data.py`; its result is `results/shared_training_data_audit_20261117.json`.

```bash
UV_CACHE_DIR=/tmp/uv-cache-p6 uv run --python 3.12 scripts/audit_shared_training_data.py
UV_CACHE_DIR=/tmp/uv-cache-p6 uv run --python 3.12 --with pytest==8.4.2 python -m pytest -q
git diff --check
```

The audit only reads the frozen data and manifests. It does not regenerate, rewrite, sample, or upload data. Remote status is obtained from locally fetched remote-tracking refs; fetch the named branch refs first if they are absent, then rerun the audit. The complete machine-readable results, including field checks and branch status, are saved alongside this report.

## Handoff note

Train and Dev are ready as frozen shared data inputs. They are not themselves a 4,000-row training subset. The held-out data remains local/ignored and is absent from the inspected GitHub trees. No B/C module was changed.
