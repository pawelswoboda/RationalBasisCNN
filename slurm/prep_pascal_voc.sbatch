#!/bin/bash
#SBATCH --job-name=voc-prep
#SBATCH --account=hpc-prf-llmrout
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# Downloads PascalVOC-Keypoints and extracts the VGG16 features once
# (experiments/prepare_pascal_voc.py). Run before pascal_voc.sbatch jobs.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/hpc/env.sh"   # --root defaults to $RBCNN_DATA_ROOT
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
# $RBCNN_DATA_ROOT is on group storage, which is READ-ONLY on compute nodes, so
# preparation writes to the scratch staging root instead.
${PYTHON:-$RBCNN_CONDA_ENV/bin/python} experiments/prepare_pascal_voc.py --root "$RBCNN_DATA_STAGE"
echo
echo "Prepared into $RBCNN_DATA_STAGE/PascalVOC."
echo "Now promote it to the canonical root FROM A LOGIN NODE:"
echo "    bash hpc/promote_dataset.sh PascalVOC"
