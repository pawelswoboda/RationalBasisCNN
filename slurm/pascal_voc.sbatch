#!/bin/bash
#SBATCH --job-name=voc
#SBATCH --account=hpc-prf-llmrout
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# One PascalVOC run (experiments/pascal_voc.py) for configuration $CONFIG
# (see configs.sh) and seed $SEED. Env: CONFIG (required), SEED (0),
# EPOCHS (15), PYTHON (python), WANDB (unset; set to 1 to log to wandb).
# Submit from the repo root, e.g. CONFIG=mv_K4 SEED=1 sbatch slurm/pascal_voc.sbatch
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR}"
source "$ROOT/hpc/env.sh"          # data, cache and wandb roots on project storage
source "$ROOT/slurm/configs.sh"
: "${CONFIG:?set CONFIG=<name>}"

# $RBCNN_DATA_ROOT is read-only on compute nodes, so PascalVOC must already be
# prepared and promoted; otherwise PyG would try to process it here and fail
# with an opaque "Read-only file system" error deep inside the data loader.
if [ ! -r "$RBCNN_DATA_ROOT/PascalVOC/Aeroplane/processed/training.pt" ]; then
  echo "ERROR: PascalVOC is not prepared at $RBCNN_DATA_ROOT" >&2
  echo "  1) sbatch slurm/prep_pascal_voc.sbatch" >&2
  echo "  2) bash hpc/promote_dataset.sh PascalVOC   # from a login node" >&2
  exit 1
fi
SEED="${SEED:-0}"; EPOCHS="${EPOCHS:-15}"; PYTHON="${PYTHON:-$RBCNN_CONDA_ENV/bin/python}"
LOG="$RBCNN_RESULTS_LOG_ROOT/pascal_voc/${CONFIG}_seed${SEED}.log"
mkdir -p "$(dirname "$LOG")" "$RBCNN_WANDB_ROOT"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" WANDB_DIR="$RBCNN_WANDB_ROOT"
WANDB_FLAGS=""; [ -n "${WANDB:-}" ] && WANDB_FLAGS="--wandb --wandb_group pascal_voc/${CONFIG} --wandb_name voc_${CONFIG}_seed${SEED}"
echo "job=${SLURM_JOB_ID} node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) config=$CONFIG seed=$SEED epochs=$EPOCHS"
cd "$ROOT"
$PYTHON experiments/pascal_voc.py $(config_args "$CONFIG") --epochs "$EPOCHS" --seed "$SEED" $WANDB_FLAGS 2>&1 | tee "$LOG"
echo DONE >> "$LOG"
