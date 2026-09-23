#!/bin/bash
#SBATCH --job-name=spair
#SBATCH --account=hpc-prf-llmrout
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# One SPair-71k run (experiments/spair71k.py) on one H100 for configuration
# $CONFIG -- the 2-D PascalVOC configurations of configs.sh apply unchanged --
# and seed $SEED. Env: CONFIG (required), SEED (0), EPOCHS (15),
# LAYOUT (large), NUM_WORKERS (cpus-per-task - 2), PYTHON (project env),
# WANDB (unset; set to 1 to log to wandb).
# Submit from the repo root, e.g. CONFIG=mv_K4 SEED=0 sbatch slurm/spair71k.sbatch
# The 2 h limit is an estimate, not a measurement: PascalVOC trains ~1,300
# pairs/s on an H100, which puts SPair-71k at ~1 min per epoch (53,340 training
# pairs plus 17,618 evaluation pairs). Tighten it after the first run; short
# limits backfill much better on gpu_h100.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR}"
source "$ROOT/hpc/env.sh"          # data, cache and wandb roots on project storage
source "$ROOT/slurm/configs.sh"
: "${CONFIG:?set CONFIG=<name>}"

# $RBCNN_DATA_ROOT is read-only on compute nodes, so SPair71k must already be
# prepared and promoted; otherwise the dataset would try to process itself here
# and fail with "Read-only file system".
if [ ! -r "$RBCNN_DATA_ROOT/SPair71k/processed/images.pt" ] || \
   [ ! -r "$RBCNN_DATA_ROOT/SPair71k/processed/pairs.pt" ]; then
  echo "ERROR: SPair71k is not prepared at $RBCNN_DATA_ROOT" >&2
  echo "  1) sbatch slurm/prep_spair71k.sbatch" >&2
  echo "  2) bash hpc/promote_dataset.sh SPair71k   # from a login node" >&2
  exit 1
fi
SEED="${SEED:-0}"; EPOCHS="${EPOCHS:-15}"; LAYOUT="${LAYOUT:-large}"
NUM_WORKERS="${NUM_WORKERS:-$(( SLURM_CPUS_PER_TASK > 2 ? SLURM_CPUS_PER_TASK - 2 : 0 ))}"
PYTHON="${PYTHON:-$RBCNN_CONDA_ENV/bin/python}"
LOGDIR=spair71k; [ "$LAYOUT" != large ] && LOGDIR="spair71k_${LAYOUT}"   # keep the benchmark table clean
LOG="$RBCNN_RESULTS_LOG_ROOT/$LOGDIR/${CONFIG}_seed${SEED}.log"
mkdir -p "$(dirname "$LOG")" "$RBCNN_WANDB_ROOT"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" WANDB_DIR="$RBCNN_WANDB_ROOT"
WANDB_FLAGS=""; [ -n "${WANDB:-}" ] && WANDB_FLAGS="--wandb --wandb_group $LOGDIR/${CONFIG} --wandb_name ${LOGDIR}_${CONFIG}_seed${SEED}"
echo "job=${SLURM_JOB_ID} node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) config=$CONFIG seed=$SEED epochs=$EPOCHS layout=$LAYOUT workers=$NUM_WORKERS"
cd "$ROOT"
$PYTHON experiments/spair71k.py $(config_args "$CONFIG") --epochs "$EPOCHS" --seed "$SEED" \
  --layout "$LAYOUT" --num_workers "$NUM_WORKERS" $WANDB_FLAGS 2>&1 | tee "$LOG"
echo DONE >> "$LOG"
