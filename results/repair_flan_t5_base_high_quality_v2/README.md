# Flan-T5-base Repair high-quality v2 Development result

The selected checkpoint uses three epochs. It was selected on the frozen
50-example Repair Development set after a matched two-epoch comparison.

| Epochs | Parse | Exact match | Repair success | Feasibility | Dev loss | Optimality |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 46/50 (92%) | 31/50 (62%) | 34/50 (68%) | 35/50 (70%) | 0.04648 | 1.0 |
| **3** | **47/50 (94%)** | **33/50 (66%)** | **38/50 (76%)** | **39/50 (78%)** | **0.04331** | **1.0** |

Both runs used the same pinned base model, data, seed, Adafactor optimizer,
learning rate `2e-4`, batch size 1, gradient accumulation 4, BF16 loading,
gradient checkpointing and greedy decoding. The selected run completed 750
optimizer steps in 506.8 seconds on one NVIDIA RTX A6000.

The conditioning probe also passed: matched mean loss was `0.04310`, mismatched
mean loss was `2.29872`, and matched loss was lower on all 50 examples.

This is a Development result. It is not an ID/OOD Test or full-pipeline result.
The two-epoch checkpoint is not archived; its report and predictions are kept
only as model-selection evidence.
