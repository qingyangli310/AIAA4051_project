# Repair v4 executor feedback: first Dev50 run

Flan-T5-base, seed 42, three epochs, 500 executor-feedback training views.
The checkpoint is stored on the existing CUDA host at
`/data/liqingyang/project6_repair_v4_20261126/results/flan_repair_v4_executor_feedback/checkpoint-final`.
Model weights SHA-256: `827e8233061be5fb087431fcf2b90642244710be3f0b96a3ad65f4339a8557c3`.

On the 50 unique Repair v4 Dev candidates: 33/50 Executor successes (66%),
46/50 Trace parses (92%), 35/50 feasible paths (70%). This is an independent
Repair Dev result, not an end-to-end Dev100 or formal ID/OOD Test result.

The training script finished training and saved the checkpoint, but its initial
reporting pass raised an optimality exception because the localized target can
be longer than the global shortest path. The scorer now obtains the actual
shortest length from the grid. The saved checkpoint was evaluated without
retraining. The archived `evaluation.json` says `repair_output_format: plan`
because the evaluator used that option to pass the already formatted v4 Trace
prompt through unchanged; scoring still parsed Trace according to each row's
`output_format: trace_only` field.

No no-feedback v4 baseline, shuffled-feedback ablation, or corrected full
pipeline result has been measured yet.
