# Member C handoff: 2026-11-15 to 2026-11-16

## Stage acceptance outcome

C's 20–50-example acceptance requirement is complete with **40 frozen ID
examples per branch** and **120 total pipeline runs**.  The batch completed
without an unhandled sample error.  The sample contains 10 examples from each
of action substitution, deletion, insertion and premature stopping.

The following parts are real:

- A's frozen November 15 ID condition;
- A's canonical Executor and its final re-execution result;
- C's pipeline, error isolation and frozen metric implementations.

The Generator replay, self-feedback branch and Repair outputs are explicitly
marked `mock=true`.  B has not supplied the November 15 Generator/Verifier
checkpoint or prediction file, and no trained Repair checkpoint exists.  The
numbers below therefore validate integration and accounting only; they are not
model-quality claims.

## Reproduce

```bash
python scripts/run_c_day15_16_smoke.py
python -m pytest -q
```

Outputs:

- `results/member_c_day15_16_smoke_logs.jsonl`: all 120 complete records;
- `results/member_c_day15_16_summary.json`: overall and per-error metrics;
- `docs/MEMBER_C_20261115_16_CASES.md`: one success and one preserved failure.

## Branch results

| Branch | Kind | Runs | Success | Feasibility | Repair success | E2E success | Mean optimality |
|---|---|---:|---:|---:|---:|---:|---:|
| no-feedback | deterministic mock | 40 | 0/40 | 27/40 | 0/40 | 0/40 | N/A |
| self-feedback | placeholder mock | 40 | 36/40 | 40/40 | 36/40 | 36/40 | 1.0 |
| executor-feedback | deterministic mock | 40 | 36/40 | 40/40 | 36/40 | 36/40 | 1.0 |

Four failures are injected deterministically in each feedback-enabled branch
(every tenth row) so failure preservation, denominators and case reporting are
exercised.  The no-feedback placeholder deliberately keeps the candidate
unchanged.  These choices make the smoke test reproducible and must not be used
as comparative experimental evidence.

## Interface resolution

A defines `valid` as movement/token feasibility, so a feasible incomplete plan
has `valid=true, success=false`.  C's Repair routing needs task validity.  The
new `ExecutorVerifierAdapter` maps A's `success` to the Verification-task
`valid` field while preserving `executor_valid`.  This closes the interface
item identified in A's November 16 handoff.

## Training sample count

- C Repair Train artifact: **1,600 condition records**, derived from **800
  corrupted task examples across 200 base plans** (no-feedback +
  executor-feedback).
- C Repair Dev artifact: 160 condition records from 80 unique base samples.
- Upstream frozen Project 6 Train split: 17,510 PPNL path records.
- Actual model training performed by C in this stage: **none**; therefore the
  number of samples consumed by an optimizer is **0**.

## Known issues and next-stage inputs

1. B must provide real Generator and Verifier checkpoints, configs, logs and
   predictions with parse failures preserved.
2. C must train or receive a Repair checkpoint before reporting real branch
   comparisons.
3. The current smoke set is ID single-error data.  A's 162 OOD double-error
   examples are frozen and audited, but model evaluation waits for checkpoints.
4. A's complete processed Held-out JSONL is intentionally not tracked.  Its
   stored hash remains in the freeze manifest; clean-clone tests validate only
   available artifacts and full local audits validate the ignored file.
