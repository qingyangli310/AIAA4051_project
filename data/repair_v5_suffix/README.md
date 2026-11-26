# Repair v5 suffix experiment data

Exploratory Dev-only successor to Repair v4. It does not replace the v4 Train,
Dev, checkpoint, or any frozen Test file. The feedback model has 500 Train
examples; paired with the existing 500-row v4 no-feedback baseline, the Repair
Train budget remains 1,000. The v5 diagnostic Dev file has 50 examples from
Dev environments. No Test rows were used to construct or select this model.

Training reasons: 236 obstacle collisions, 88 goal-not-reached, 88 out-of-bounds,
and 88 invalid-action-token. Thirty-seven collision examples come from archived
Generator predictions on Train environments. Targets contain only the suffix
from the last valid position, while `target_plan` records the successful
prefix-plus-suffix path for scoring. See `manifest.json` for hashes.

This v5 Dev set is an exploratory alternative to the earlier 50-candidate
Repair Dev view, so do not describe the union of both as the frozen 50-example
final Dev budget.
