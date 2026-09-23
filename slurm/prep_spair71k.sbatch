#!/bin/bash
#SBATCH --job-name=spair-prep
#SBATCH --account=hpc-prf-llmrout
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# Builds the SPair-71k keypoint graphs once (experiments/prepare_spair71k.py):
# VGG16 features for all 1,800 images plus the pair tables of both layouts.
# Needs the raw tree from hpc/download_spair71k.sh. Run it before any
# spair71k.sbatch job, then promote the result from a login node.
# Submit from the repo root: sbatch slurm/prep_spair71k.sbatch
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/hpc/env.sh"
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1

if [ -r "$RBCNN_DATA_ROOT/SPair71k/processed/pairs.pt" ]; then
  echo "SPair71k is already prepared at $RBCNN_DATA_ROOT/SPair71k; nothing to do."
  exit 0
fi
if [ ! -d "$RBCNN_SPAIR_ROOT/PairAnnotation" ]; then
  echo "ERROR: raw SPair-71k not found at $RBCNN_SPAIR_ROOT" >&2
  echo "  run on a login node first: bash hpc/download_spair71k.sh" >&2
  exit 1
fi

# The raw tree is only READ from group storage, which is read-only here; the
# processed graphs are WRITTEN to the scratch staging root.
${PYTHON:-$RBCNN_CONDA_ENV/bin/python} experiments/prepare_spair71k.py \
  --root "$RBCNN_DATA_STAGE" --spair_root "$RBCNN_SPAIR_ROOT"
echo
echo "Prepared into $RBCNN_DATA_STAGE/SPair71k."
echo "Now promote it to the canonical root FROM A LOGIN NODE:"
echo "    bash hpc/promote_dataset.sh SPair71k"
