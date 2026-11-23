# Verification structured state tracking v1

This dataset preserves all rows, labels, splits, non-overlap guarantees, and
SHA-traceable provenance from `verification_high_quality_v1`. It changes only
the input representation by appending attempted coordinates derived from the
candidate actions.

The trace is label-free: it does not use the executor result and does not state
whether the plan is valid, the failure reason, or the first failing step. The
model must compare the attempted coordinates with the boundary, obstacles, and
goal. This is the structured state-tracking improvement permitted by the
Project 6 specification.

Rebuild with:

```bash
python scripts/build_state_tracking_verification_data.py
```
