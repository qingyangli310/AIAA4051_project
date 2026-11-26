# Locked Repair v4 formal OOD Test

Frozen double-error OOD candidates: 162/162 completed, zero runtime errors.
Dataset candidates were read from the frozen file; Generator was not called.
The configuration was locked before Test inference in
`configs/final_pipeline_v4_locked.yaml`.

| Condition | Success |
| --- | ---: |
| Direct candidate | 0/162 |
| Independent no-feedback Repair | 84/162 (51.9%) |
| Self feedback Repair | 27/162 (16.7%) |
| Correct executor feedback Repair | 56/162 (34.6%) |
| Shuffled executor feedback Repair | 56/162 (34.6%) |

Every OOD candidate's first executor failure was goal_not_reached. The
existing shuffled-feedback routine could not change the step/reason content
for any OOD row (0/162 changed), so its identical result is not an informative
feedback-content ablation on this split. The independent no-feedback model
outperformed the feedback model. Test results were not used to tune the model.

See `report.json` and `records.jsonl` for metrics and per-example evidence.
