# Supplemental paired ID/OOD diagnostics

These 162 matched ID/OOD pairs reuse environments from the already inspected
held-out split. They are **not a new blind Test set**. The original frozen
ID/OOD files and their formal results remain unchanged.

Each ID row has one action corruption. Each OOD row has two different action
corruptions at distinct positions. Both OOD edits independently fail, their
sequential replay produces the recorded candidate, and removing either edit
changes the final Executor trace or position. All labels were recomputed with
the canonical Executor, all clean reference paths succeed, and there is zero
Train/Dev environment overlap.

The resulting Executor reasons are balanced within each condition: 54
`goal_not_reached`, 54 `obstacle_collision`, and 54 `out_of_bounds`. The four ID
corruption kinds have counts 41, 42, 39, and 40. Under the pipeline's
same-length shuffled feedback rule, step/reason content changes on 145/162 ID
and 162/162 OOD examples. This makes feedback-content comparisons diagnostic
on both splits, unlike the original OOD set's 0/162 change.

Build reproducibly with `python scripts/build_supplemental_id_ood.py`. Do not
select or retrain a model using these already exposed environments and then
present its result here as a fresh blind evaluation.
