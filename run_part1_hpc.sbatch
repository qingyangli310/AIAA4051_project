#!/bin/bash
#SBATCH --job-name=project6_part1
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "${PROJECT6_CONDA_ENV:?set PROJECT6_CONDA_ENV to the frozen environment}"

export HF_HUB_CACHE="${HF_HUB_CACHE:?set HF_HUB_CACHE to the shared model cache}"
export TOKENIZERS_PARALLELISM=false

RUN_ROOT="outputs/part1_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" logs

python scripts/check_environment.py
python scripts/build_part1_data.py
python scripts/run_part1_symbolic.py

python scripts/train_part1.py \
  --task generation --model t5-small \
  --output "$RUN_ROOT/generation_t5_small"
python scripts/train_part1.py \
  --task generation --model bart-base \
  --output "$RUN_ROOT/generation_bart_base"

python scripts/compare_part1_models.py \
  "$RUN_ROOT/generation_t5_small/run_report.json" \
  "$RUN_ROOT/generation_bart_base/run_report.json" \
  --output "$RUN_ROOT/model_selection.json"

PRIMARY=$(python - "$RUN_ROOT/model_selection.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))["primary_model"]
if value is None:
    raise SystemExit("Both Generation baselines collapsed; primary model selection blocked")
print(value)
PY
)

case "$PRIMARY" in
  t5-small) PRIMARY_DIR="t5_small" ;;
  bart-base) PRIMARY_DIR="bart_base" ;;
  *) echo "Unsupported primary model: $PRIMARY" >&2; exit 2 ;;
esac

python scripts/train_part1.py \
  --task verification --model "$PRIMARY" \
  --output "$RUN_ROOT/verification_${PRIMARY_DIR}"
python scripts/train_part1.py \
  --task repair --model "$PRIMARY" \
  --output "$RUN_ROOT/repair_${PRIMARY_DIR}"

for MODEL_DIR in t5_small bart_base; do
  python scripts/evaluate_part1_checkpoint.py \
    --task generation \
    --checkpoint "$RUN_ROOT/generation_${MODEL_DIR}/checkpoint-final" \
    --data data/part1/generation_id_test.jsonl \
    --output "$RUN_ROOT/generation_${MODEL_DIR}_id"
done

python scripts/evaluate_part1_checkpoint.py \
  --task verification \
  --checkpoint "$RUN_ROOT/verification_${PRIMARY_DIR}/checkpoint-final" \
  --data data/part1/verification_id_test.jsonl \
  --output "$RUN_ROOT/verification_${PRIMARY_DIR}_id"
python scripts/evaluate_part1_checkpoint.py \
  --task verification \
  --checkpoint "$RUN_ROOT/verification_${PRIMARY_DIR}/checkpoint-final" \
  --data data/part1/verification_ood_test.jsonl \
  --output "$RUN_ROOT/verification_${PRIMARY_DIR}_ood"
python scripts/evaluate_part1_checkpoint.py \
  --task repair \
  --checkpoint "$RUN_ROOT/repair_${PRIMARY_DIR}/checkpoint-final" \
  --data data/part1/repair_id_test.jsonl \
  --output "$RUN_ROOT/repair_${PRIMARY_DIR}_id"
python scripts/evaluate_part1_checkpoint.py \
  --task repair \
  --checkpoint "$RUN_ROOT/repair_${PRIMARY_DIR}/checkpoint-final" \
  --data data/part1/repair_ood_test.jsonl \
  --output "$RUN_ROOT/repair_${PRIMARY_DIR}_ood"

echo "Part 1 run completed: $RUN_ROOT"
