# Locked Repair v4 formal ID Test

Frozen single-error ID candidates: 162/162 completed, zero runtime errors.
Dataset candidates were read from the frozen file; Generator was not called.
The configuration was locked before Test inference in
`configs/final_pipeline_v4_locked.yaml`.

| Condition | Success |
| --- | ---: |
| Direct candidate | 0/162 |
| Independent no-feedback Repair | 98/162 (60.5%) |
| Self feedback Repair | 48/162 (29.6%) |
| Correct executor feedback Repair | 89/162 (54.9%) |
| Shuffled executor feedback Repair | 88/162 (54.3%) |

The correct executor feedback model did not outperform the independent
no-feedback model. The shuffled feedback changed step/reason content on 125
of 162 candidates and produced nearly the same result as correct feedback.
Candidate failure reasons were 119 goal_not_reached, 28 obstacle_collision,
and 15 out_of_bounds. These results do not support a general feedback benefit
on formal ID candidates. Test results were not used to tune the model.

See `report.json` and `records.jsonl` for metrics and per-example evidence.
