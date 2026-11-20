#!/bin/bash
#SBATCH --job-name=gen_trace
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/gen_trace_%j.out
#SBATCH --error=logs/gen_trace_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from day14_hpc_v1}"
MODEL_DIR="${GEN_MODEL_DIR:-}"
if [[ -z "$MODEL_DIR" ]]; then
    for candidate in models/flan-t5-base generation_models_v5/models/flan-t5-base generation_models_v4/models/flan-t5-base generation_models_v3/models/flan-t5-base dev_evaluation_v6/models/flan-t5-base; do
        if [[ -f "$candidate/model_manifest.json" ]]; then
            MODEL_DIR="$candidate"
            break
        fi
    done
fi
if [[ ! -f "$MODEL_DIR/model_manifest.json" ]]; then
    echo 'Missing pretrained Flan model; set GEN_MODEL_DIR to its directory.' >&2
    exit 2
fi
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "${GEN_CONDA_ENV:-/data/user/$USER/envs/day14_env}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python -u generation_trace_v9/train_trace.py --model-dir "$MODEL_DIR" \
    --output "outputs/trace_v9_trace_only_${SLURM_JOB_ID}"
