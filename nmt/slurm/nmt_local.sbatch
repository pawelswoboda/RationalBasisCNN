#!/bin/bash
#SBATCH --job-name=nmt-local
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# One NMT run on the group's local Slurm cluster (RTX 4090 / 5090 nodes), the
# counterpart of nmt_voc.sbatch / nmt_spair.sbatch for PC2. Env: CONFIG
# (required), PYTHON (default: the project venv). Submit from nmt/:
#   CONFIG=runs/local/voc_spline_seed0.json sbatch slurm/nmt_local.sbatch
# The run log (stdout + stderr) also goes to runs/local/logs/<config name>.log,
# ending with a DONE line.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
: "${CONFIG:?set CONFIG=<path/to/config.json>}"
PYTHON="${PYTHON:-/home/pswoboda/.venv/bin/python}"
NAME=$(basename "$CONFIG" .json)
LOG="runs/local/logs/${NAME}.log"
mkdir -p runs/local/logs slurm/logs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export WANDB_MODE="${WANDB:-disabled}"
# Same basis evaluation as the logged PC2 runs (see hpc/env.sh).
export RATIONAL_BASIS_IMPL="${RATIONAL_BASIS_IMPL:-eager}"
PORT=$($PYTHON -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "job=${SLURM_JOB_ID} node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) config=$CONFIG port=$PORT" | tee "$LOG"
$PYTHON -m torch.distributed.run --nproc_per_node=1 --master_port="$PORT" train_eval.py "$CONFIG" 2>&1 | tee -a "$LOG"
echo DONE >> "$LOG"
