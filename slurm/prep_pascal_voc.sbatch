#!/bin/bash
#SBATCH --job-name=voc-prep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# Downloads PascalVOC-Keypoints and extracts the VGG16 features once
# (experiments/prepare_pascal_voc.py). Run before pascal_voc.sbatch jobs.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
${PYTHON:-python} experiments/prepare_pascal_voc.py
