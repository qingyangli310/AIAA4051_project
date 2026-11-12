# Experiment Protocol

## Frozen interface rules (2026-11-12)

- Grid coordinates and action error steps are 0-based.
- The first action is step `0`; the second action is step `1`.
- `goal_not_reached` uses `first_error_step = len(plan)`. This marks the end of the path and is not an additional executed action.
- Environments are split into Train, Dev, and Held-out Test by `environment_id`.
- Train, Dev, and Held-out Test must be mutually disjoint.
- ID single-error and OOD double-error are paired conditions derived from the same Held-out Test environments and linked by `base_sample_id`.

## Dataset composition

| Environment split or condition | Required coverage |
|---|---|
| Train | Generation, Verification, and Repair training examples |
| Dev | All three tasks for model selection and parameter tuning |
| Held-out Test | A protected environment pool used only for final evaluation conditions |
| ID single-error condition | Valid, feasible non-optimal, and single-error candidates from Held-out Test |
| OOD double-error condition | Two independently introduced errors on the same paired Held-out Test base samples |

Valid examples must include both optimal paths and feasible but non-optimal paths. A longer feasible path is valid; path quality is measured separately by Optimality.

## Frozen metrics

- **Success Rate:** successful final plans divided by all evaluated plans.
- **Feasibility:** plans with no collision, boundary, or invalid-token violation divided by all plans. A plan that ends without reaching the goal is feasible but not successful.
- **Optimality:** `optimal_length / predicted_length`, computed only for successful plans. The score is `1.0` for an optimal path and below `1.0` for a longer successful path.
- **Repair Success Rate:** repaired plans that succeed divided by invalid candidate plans sent to Repair.
- **End-to-End Success Rate:** environments whose final pipeline result succeeds divided by all evaluated environments.
