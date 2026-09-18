#!/bin/bash
#SBATCH --job-name=ncaltech101
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# One N-Caltech101 (AEGNN) run (experiments/ncaltech101.py) for configuration $CONFIG (see
# configs.sh) and seed $SEED. Env: CONFIG (required), SEED (0), EPOCHS (30),
# PYTHON (python), WANDB (unset; set to 1 to log to wandb).
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR}"
source "$ROOT/slurm/configs.sh"
: "${CONFIG:?set CONFIG=<name>}"
SEED="${SEED:-0}"; EPOCHS="${EPOCHS:-30}"; PYTHON="${PYTHON:-python}"
LOG="$ROOT/results/logs/ncaltech101/${CONFIG}_seed${SEED}.log"
mkdir -p "$(dirname "$LOG")" "$ROOT/wandb"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" WANDB_DIR="$ROOT/wandb"
WANDB_FLAGS=""; [ -n "${WANDB:-}" ] && WANDB_FLAGS="--wandb --wandb_group ncaltech101/${CONFIG} --wandb_name ncaltech101_${CONFIG}_seed${SEED}"
echo "job=${SLURM_JOB_ID} node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) config=$CONFIG seed=$SEED epochs=$EPOCHS"
cd "$ROOT"
$PYTHON experiments/ncaltech101.py $(config_args "$CONFIG") --epochs "$EPOCHS" --seed "$SEED" $WANDB_FLAGS 2>&1 | tee "$LOG"
echo DONE >> "$LOG"
