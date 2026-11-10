# AIAA4051 Project 6

Prompting and Fine-tuning Language Models for Grid Path Planning.

This repository implements a single-goal grid-navigation research pipeline:

```text
Generate → Verify → Repair → Execute → Evaluate
```

Baseline 1 currently uses `google-t5/t5-small`. If the project manager later changes the decision to Flan-T5, the team will update the shared configuration and protocol in one coordinated change before running new formal experiments.

## Repository Structure

```text
.
├── configs/                Shared experiment configuration
├── data/
│   ├── raw/                Local source data; not committed
│   ├── processed/          Generated data; not committed
│   └── manifests/          Versioned split and construction manifests
├── docs/                   Team protocol, project explanation, and task plans
├── src/
│   ├── executor/           Symbolic grid executor
│   ├── data/               Data construction and corruption
│   ├── models/             Shared model loading and generation
│   ├── verifier/           Plan verification
│   ├── repair/             Plan repair
│   └── evaluation/         Metrics and error analysis
├── scripts/                Runnable entry points and environment checks
├── tests/                  Automated tests
├── results/                Versioned summaries; large run outputs are ignored
├── poster/                 Poster sources
├── report/                 Final report sources
├── requirements-project6.txt
└── README.md
```

## Shared Environment

The frozen development environment is:

```text
Python 3.11.9
PyTorch 2.9.1
CUDA 12.6 wheel
Transformers 4.57.6
NVIDIA RTX 4090 24 GB for formal experiments
Seed 42
```

Create the environment:

```bash
conda create -n project6 python=3.11.9 -y
conda activate project6

python -m pip install --upgrade pip
python -m pip install torch==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-project6.txt
```

Check the environment:

```bash
python scripts/check_environment.py
```

CPU-only machines may run the executor, data construction, unit tests, and small smoke tests. Formal model comparisons and final evaluation must run on the same RTX 4090.

## Frozen Protocol

Read these documents before implementation:

- `docs/Project6_团队统一协议.md`
- `docs/Project6_项目说明与三人分工.md`
- `docs/Project6_11月10日至12日任务清单.md`

The shared experiment configuration is stored in `configs/base.yaml`.

## Team Ownership

| Member | Ownership | Reviewer |
|---|---|---|
| A | Data, executor, corruption, environment split | C |
| B | Generation, verification, model training | A |
| C | Repair, end-to-end pipeline, evaluation | B |

Every critical module follows the rule `one author + one reviewer`.

## Git Workflow

Branches:

```text
main
develop
feature/a-executor
feature/b-verifier
feature/c-repair-pipeline
```

Rules:

- Do not push directly to `main`.
- Open a Pull Request for each logical change.
- Require at least one review before merging.
- Use Squash and merge.
- Run relevant tests before merging.
- Do not commit model checkpoints, secrets, local absolute paths, or generated raw data.

## Immediate Milestone

By 2026-11-12:

```text
A: executor toy example
B: model inference smoke test
C: mock generate-verify-repair pipeline
```

Baseline 1 formal experiments use `google-t5/t5-small` unless a later recorded team decision switches the project to Flan-T5.
