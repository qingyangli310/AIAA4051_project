# Project 6 Part 1 implementation and runbook

## Scope and status

Part 1 requires a correct single-goal Executor and candidate constructor,
direct Generation and Verification or Repair baselines, and a matched
comparison of at least two encoder-decoder models on the same 6 by 6 ID data.

The repository contains the complete data, symbolic reference acceptance, model
training, checkpoint reload, diagnostics, evaluation and development-only model
selection workflow. Protocol v2 adds an explicit ASCII grid and relative goal
offset to every prompt and balances the Generation training set to 250 examples
per first action. These changes address the observed output-prior collapse.

## Frozen data budget

| Task | Train | Dev | ID test | OOD test |
|---|---:|---:|---:|---:|
| Generation | 1,000 | 20 | 162 | Not applicable |
| Verification | 1,500 | 90 | 200, balanced valid and invalid | 162 double-error candidates |
| Repair | 1,500 | 90 | 162 single-error candidates | 162 double-error candidates |
| Total | 4,000 | 200 | Task-specific caps respected | Task-specific caps respected |

All sets use seed 42.  Train, Dev and Held-out Test environment intersections
are zero.  The manifest records every file count and SHA-256 value in
`data/part1/manifest.json`.

Generation uses the task-specific post-diagnostic defaults: learning rate
`2e-4`, per-device batch size 4 and no gradient accumulation. This yields about
750 updates over three epochs without increasing the number of sample
presentations. Verification and Repair retain learning rate `5e-5`, batch size
8 and gradient accumulation 2.

## Baseline protocol

The current repository protocol explicitly uses `google-t5/t5-small` and
`facebook/bart-base`.  T5-small is not described as Flan-T5.  If the project
manager chooses Flan-T5, the model identifier, revision and all affected runs
must be changed together.

The two Generation models use identical data, seed, batch size, effective batch
size, three-epoch cap, optimizer and decoding.  Development ranking uses, in
order, Success Rate, Feasibility and mean Optimality.  Test results are never
used for selection.  If both models have zero Development success and at least
90 percent of outputs are identical, selection stops with
`blocked_by_generation_collapse`.

After selecting the primary model, only that model runs the complete
Verification and Repair study, as allowed by the project specification.  Both
Generation models are then evaluated on the same ID set for the required
two-model comparison.

## Collapse diagnostics

Each formal run records:

- exact model name and immutable revision;
- Train and Dev SHA-256 values;
- input and target token lengths;
- optimizer steps, loss history, device, BF16 status and runtime;
- a selected parameter signature before and after training;
- reloaded checkpoint predictions;
- unique prediction count and most-common prediction share;
- matched-input and mismatched-input teacher-forced loss.

This directly covers the unresolved checks from the November 15 debugging
package.  Token targets use the tokenizer's `text_target` path explicitly;
label padding is fixed to `-100`; the final checkpoint is saved, reloaded and
used for every reported prediction.

## Commands

CPU-side acceptance:

```bash
python scripts/build_part1_data.py
python scripts/run_part1_symbolic.py
python -m pytest -q
```

Before a formal Generation rerun, verify learnability on 32 examples:

```bash
python scripts/train_part1.py \
  --task generation --model bart-base \
  --diagnostic-train-examples 32 --epochs 30 \
  --output outputs/overfit_bart_32
```

This diagnostic is marked `formal_part1_run=false` and must not be reported as
a formal baseline.

Formal HPC run:

```bash
export PROJECT6_CONDA_ENV=/data/user/$USER/envs/project6
export HF_HUB_CACHE=/data/user/$USER/hf_hub_cache
sbatch run_part1_hpc.sbatch
```

The symbolic result is an implementation upper bound and must never be placed
in a table as a T5 or BART result.

## Acceptance criteria

Part 1 is ready for submission when:

1. all repository tests pass;
2. the symbolic acceptance report passes;
3. both formal Generation runs complete on matched data and budgets;
4. the primary model is selected from Development results without collapse;
5. primary Verification and Repair checkpoints complete;
6. ID test results exist for both Generation models;
7. primary ID and OOD Verification and Repair results exist;
8. every reported table links to the corresponding run report and data hash.
