# Repair v3 deduplicated-data run

CUDA retraining used the 1500-row deduplicated successor dataset at
`data/repair_generation_like_trace_hardonly_v3_dedup/repair_train.jsonl`.
The run starts from the pinned T5-small base snapshot rather than continuing
from the previous Repair checkpoint.

## Result

| Metric | Deduplicated run | Previous v3 run |
|---|---:|---:|
| Parse rate | 73/90 (81.1%) | 71/90 (78.9%) |
| Feasible | 51/90 (56.7%) | 55/90 (61.1%) |
| Successful repair | 48/90 (53.3%) | 54/90 (60.0%) |
| Mean optimality | 0.9833 | 0.9010 |

The cleaned dataset removes the two duplicate complete examples, but this
single seeded run does not improve total Dev success. It produces more nearly
optimal paths when it does succeed. The matched/mismatched conditioning losses
are 0.1380 and 2.1071 respectively, providing evidence that the checkpoint is
using its repair input rather than ignoring it.

The full 242 MB checkpoint is intentionally not committed to ordinary Git.
`run_report.json` records the exact model revision, data hashes, configuration,
metrics, and checkpoint hash used for the external artifact.

