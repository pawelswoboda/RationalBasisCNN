#!/bin/bash
#SBATCH --job-name=faust
#SBATCH --account=hpc-prf-llmrout
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# One FAUST run (experiments/faust.py) for configuration $CONFIG (see
# configs.sh) and seed $SEED. Env: CONFIG (required), SEED (0), EPOCHS (100),
# PYTHON (python), WANDB (unset; set to 1 to log to wandb).
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR}"
source "$ROOT/hpc/env.sh"          # data, cache and wandb roots on project storage
source "$ROOT/slurm/configs.sh"
: "${CONFIG:?set CONFIG=<name>}"

# $RBCNN_DATA_ROOT is read-only on compute nodes, so FAUST must already be
# prepared and promoted; otherwise PyG would try to process it here and fail
# with an opaque "Read-only file system" error deep inside the data loader.
if [ ! -r "$RBCNN_DATA_ROOT/FAUST/processed/training.pt" ]; then
  echo "ERROR: FAUST is not prepared at $RBCNN_DATA_ROOT" >&2
  echo "  1) python experiments/prepare_faust.py --src <MPI-FAUST.zip> --root "$RBCNN_DATA_STAGE"" >&2
  echo "  2) bash hpc/promote_dataset.sh FAUST   # from a login node" >&2
  exit 1
fi
SEED="${SEED:-0}"; EPOCHS="${EPOCHS:-100}"; PYTHON="${PYTHON:-$RBCNN_CONDA_ENV/bin/python}"
LOG="$RBCNN_RESULTS_LOG_ROOT/faust/${CONFIG}_seed${SEED}.log"
mkdir -p "$(dirname "$LOG")" "$RBCNN_WANDB_ROOT"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" WANDB_DIR="$RBCNN_WANDB_ROOT"
WANDB_FLAGS=""; [ -n "${WANDB:-}" ] && WANDB_FLAGS="--wandb --wandb_group faust/${CONFIG} --wandb_name faust_${CONFIG}_seed${SEED}"
echo "job=${SLURM_JOB_ID} node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) config=$CONFIG seed=$SEED epochs=$EPOCHS"
cd "$ROOT"
$PYTHON experiments/faust.py $(config_args "$CONFIG") --epochs "$EPOCHS" --seed "$SEED" $WANDB_FLAGS 2>&1 | tee "$LOG"
echo DONE >> "$LOG"
