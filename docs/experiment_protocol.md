# Experiment Protocol

## Frozen interface rules (2026-11-12)

- Grid coordinates and action error steps are 0-based.
- The first action is step `0`; the second action is step `1`.
- `goal_not_reached` uses `first_error_step = len(plan)`. This marks the end of the path and is not an additional executed action.
- All splits are assigned by `environment_id`; environments must never cross splits.

## Dataset composition

| Split | Required coverage |
|---|---|
| Train | Generation, Verification, and Repair training examples |
| Dev | All three tasks for model selection and parameter tuning |
| ID Test | Valid paths and single-error scenarios |
| OOD Test | Double-error Verification and Repair generalization |

## Proposed metrics for meeting confirmation

- **Success Rate:** successful final plans divided by all evaluated plans.
- **Feasibility (proposed):** plans with no collision, boundary, or invalid-token violation divided by all plans. The team still needs to confirm whether a path that ends without reaching the goal counts as feasible.
- **Optimality (proposed):** `predicted_length / optimal_length`. The team still needs to confirm whether this is computed only for successful plans.
- **Repair Success Rate:** repaired plans that succeed divided by invalid candidate plans sent to Repair.
- **End-to-End Success Rate:** environments whose final pipeline result succeeds divided by all evaluated environments.
