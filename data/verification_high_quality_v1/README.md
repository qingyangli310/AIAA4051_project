# Verification high-quality v1

This is the frozen Verification dataset for the final 4,000-example protocol.
It contributes 1,000 Train rows and 50 Dev rows alongside Generator v9
(2,000/100) and Repair high-quality v1 (1,000/50 for the formal Dev budget).

Quality guarantees enforced by the builder and tests:

- Train is exactly 500 valid and 500 invalid examples.
- Valid examples are exactly balanced between shortest and feasible non-optimal
  plans.
- Invalid Train examples are exactly balanced across `goal_not_reached`,
  `obstacle_collision`, `out_of_bounds`, and `invalid_action_token`.
- Every label, first-error step, and failure reason is recomputed with the
  canonical executor.
- Every row has a unique `base_sample_id` and unique model input/target pair.
- Train and Dev environments are disjoint.
- Base sample IDs and semantic `(environment_id, candidate_plan)` pairs have
  zero overlap with Generator v9 and Repair high-quality v1.

Generator v9 uses all 572 Train environments, so environment-level separation
between tasks is mathematically impossible without changing the frozen
Generator data. Cross-task separation is therefore enforced at the base-sample
and semantic-plan levels. Environment-level separation remains mandatory and
zero-overlap between Train, Dev, and Held-out Test splits.

Rebuild deterministically with:

```bash
python scripts/build_high_quality_verification_data.py
```

The exact counts, source files, overlap checks, and SHA-256 values are recorded
in `manifest.json`.
