#!/bin/bash
#SBATCH --job-name=faust
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# One FAUST run (experiments/faust.py) for configuration $CONFIG (see
# configs.sh) and seed $SEED. Env: CONFIG (required), SEED (0), EPOCHS (100),
# PYTHON (python), WANDB (unset; set to 1 to log to wandb).
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR}"
source "$ROOT/slurm/configs.sh"
: "${CONFIG:?set CONFIG=<name>}"
SEED="${SEED:-0}"; EPOCHS="${EPOCHS:-100}"; PYTHON="${PYTHON:-python}"
LOG="$ROOT/results/logs/faust/${CONFIG}_seed${SEED}.log"
mkdir -p "$(dirname "$LOG")" "$ROOT/wandb"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" WANDB_DIR="$ROOT/wandb"
WANDB_FLAGS=""; [ -n "${WANDB:-}" ] && WANDB_FLAGS="--wandb --wandb_group faust/${CONFIG} --wandb_name faust_${CONFIG}_seed${SEED}"
echo "job=${SLURM_JOB_ID} node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) config=$CONFIG seed=$SEED epochs=$EPOCHS"
cd "$ROOT"
$PYTHON experiments/faust.py $(config_args "$CONFIG") --epochs "$EPOCHS" --seed "$SEED" $WANDB_FLAGS 2>&1 | tee "$LOG"
echo DONE >> "$LOG"
