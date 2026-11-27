# Supplemental ID/OOD Repair diagnostics

These runs evaluate the selected Repair v4 method and the exploratory Repair v5
suffix method on `data/supplemental_id_ood_v1`.  The dataset contains 162 paired
ID/OOD candidates.  Each split has 54 final `goal_not_reached`, 54
`obstacle_collision`, and 54 `out_of_bounds` cases.

This is a **supplemental diagnostic**, not a new blind Test.  The construction
uses environments whose earlier held-out results had already been inspected.
It must not replace or be merged with the frozen formal Test results.

All four runs completed 162/162 rows with zero runtime errors.

| Method / condition | Supplemental ID | Supplemental OOD |
|---|---:|---:|
| Independent no-feedback v4 | 87/162 (53.7%) | 76/162 (46.9%) |
| Verifier self-feedback + v4 | 37/162 (22.8%) | 30/162 (18.5%) |
| Correct Executor feedback + v4 | 80/162 (49.4%) | 57/162 (35.2%) |
| Shuffled Executor feedback + v4 | 69/162 (42.6%) | 51/162 (31.5%) |
| Correct Executor feedback + v5 suffix | 17/162 (10.5%) | 10/162 (6.2%) |
| Shuffled Executor feedback + v5 suffix | 6/162 (3.7%) | 2/162 (1.2%) |

The correct-vs-shuffled gaps show that both feedback models respond to feedback
content on this corrected diagnostic.  However, the selected v4 feedback model
still trails the independently trained no-feedback v4 model, and the v5 suffix
method is substantially worse than v4.  Therefore this experiment does not
support selecting v5 or claiming that Executor feedback is the best overall
Repair condition.

The archived reports and per-example records are in:

- `results/supplemental_v4_id/`
- `results/supplemental_v4_ood/`
- `results/supplemental_v5_id/`
- `results/supplemental_v5_ood/`

