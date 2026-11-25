# Repair feedback-localized v4 data

This directory contains the budget-compliant data views for Repair v4.

- `no_feedback/`: 500 Train and 50 paired Dev views without error location or
  reason.
- `executor_feedback/`: the same candidates with authoritative error step,
  reason, valid prefix, and resume position.
- `manifest.json`: counts, hashes, split isolation, and the budget accounting.

The 1,000 materialized Train rows use 500 unique candidates under two matched
conditions.  The two Dev files are views of the same 50 unique candidates and
are used only for paired method selection.  All targets preserve the valid
prefix and replace the failed suffix with an executable route to the goal.

Rebuild instructions and training commands are in
`docs/REPAIR_V4_RUNBOOK.md`.
