#!/bin/bash
#SBATCH --job-name=hpatches
#SBATCH --account=hpc-prf-llmrout
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# Out-of-task transfer on HPatches for one NMT configuration, all its seeds
# (experiments/hpatches_transfer.py). Env: CONFIG (required, an NMT config
# name such as spline or mv_K4), SEEDS ("0 1 2 3 4"), NUM_KEYPOINTS (128),
# DISTRACTORS (1.0), NMT_RUNS (the NMT run root), PYTHON.
# Submit from the repo root: CONFIG=spline sbatch slurm/hpatches.sbatch
#
# ~2.5 min per run, so five seeds fit comfortably in the hour.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR}"
source "$ROOT/hpc/env.sh"
: "${CONFIG:?set CONFIG=<nmt config name>}"
cd "$ROOT"

NMT_RUNS="${NMT_RUNS:-/scratch/hpc-prf-llmrout/hpcabpo/nmt/runs/spair_vgg16}"
SEEDS=(${SEEDS:-0 1 2 3 4})
NUM_KEYPOINTS="${NUM_KEYPOINTS:-128}"; DISTRACTORS="${DISTRACTORS:-1.0}"
# keep sweeps with different settings in separate directories
OUT="$RBCNN_PFS_ROOT/hpatches/k${NUM_KEYPOINTS}_d${DISTRACTORS}"
PYTHON="${PYTHON:-$RBCNN_CONDA_ENV/bin/python}"
mkdir -p "$OUT"

if [ ! -d "$RBCNN_DATA_ROOT/HPatches" ]; then
  echo "ERROR: HPatches not found at $RBCNN_DATA_ROOT/HPatches" >&2
  echo "  run: bash hpc/download_hpatches.sh   (login node)" >&2
  exit 1
fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "job=${SLURM_JOB_ID} node=$(hostname) config=$CONFIG seeds=${SEEDS[*]}"

for s in "${SEEDS[@]}"; do
  RUN="$NMT_RUNS/${CONFIG}_seed${s}"
  if [ ! -r "$RUN/settings.json" ]; then
    echo "skipping $RUN (no settings.json)" >&2
    continue
  fi
  echo "=== $CONFIG seed $s ==="
  $PYTHON experiments/hpatches_transfer.py \
    --run_dir "$RUN" \
    --num_keypoints "$NUM_KEYPOINTS" \
    --distractors "$DISTRACTORS" \
    --out "$OUT/${CONFIG}_seed${s}.json"
done
