# Repair high-quality v1 result

T5-small was trained from the pinned base revision for three epochs on the
validated 1,000-row high-quality dataset. Evaluation uses the same frozen
90-row Repair Dev set as the previous size sweep.

| Dataset | Repair success | Parse rate | Exact match | Mean optimality |
|---|---:|---:|---:|---:|
| high-quality v1, 1,000 | **64/90 (71.1%)** | **80/90 (88.9%)** | **62/90 (68.9%)** | **1.0000** |
| hard-only nested, 1,000 | 55/90 (61.1%) | 70/90 (77.8%) | 23/90 (25.6%) | 0.9582 |
| hard-only deduplicated, 1,500 | 48/90 (53.3%) | 73/90 (81.1%) | 22/90 (24.4%) | 0.9833 |

The controlled 1,000-vs-1,000 comparison uses the same model, seed, epochs,
learning rate, batch size, decoding, and Dev set. The high-quality mix improves
Repair success by 9/90 and exact match by 39/90 over the hard-only subset.

This is still a single-seed Dev result. Freeze the dataset and checkpoint before
the one-time held-out ID/OOD evaluation.

