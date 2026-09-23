#!/usr/bin/env bash
# Creates or updates the project conda environment on group storage and runs
# the unit suite. Does not submit any Slurm job. Run on a login node.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "$RBCNN_REPO_ROOT"

mkdir -p \
  "$RBCNN_DATA_ROOT" "$RBCNN_DATA_STAGE" "$RBCNN_DOWNLOAD_ROOT" "$RBCNN_CACHE_ROOT" \
  "$RBCNN_WANDB_ROOT" "$RBCNN_PFS_ROOT/tmp" "$RBCNN_SLURM_LOG_ROOT" \
  "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS"

module load lang/Miniforge3/25.3.0-3
if [[ -x "$RBCNN_CONDA_ENV/bin/python" ]]; then
  conda env update --prefix "$RBCNN_CONDA_ENV" --file environment-hpc.yml --prune -y
else
  mamba env create --prefix "$RBCNN_CONDA_ENV" --file environment-hpc.yml -y
fi
"$RBCNN_CONDA_ENV/bin/python" -m pip install --no-deps -e .
"$RBCNN_CONDA_ENV/bin/python" -m pytest
