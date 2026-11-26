# Verifier natural collision augmentation: Dev ablation

This experiment replaced 37 synthetic obstacle collisions in the 1,000-row
Verifier Train set with 37 actual Generator failures from Train environments.
The 50-row Verifier Dev set and class counts remained unchanged. Both Repair
checkpoints and the Generator remained fixed.

The new Verifier reached Macro-F1 0.567 on its independent Dev50. On the same
full Dev100 pipeline, Self feedback stayed at 66/100 and its Verifier Macro-F1
was 0.391. Executor feedback remained 78/100; independent no-feedback was
68/100 and shuffled feedback 71/100. All 100 rows completed without errors.

This augmentation did not improve Self feedback, so the new Verifier is
rejected for the final configuration. The prior Verifier checkpoint remains
selected. See the archived report and per-example records for the diagnostic
evidence. No frozen ID/OOD Test data was used for selection.
