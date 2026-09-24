#!/usr/bin/env bash
# Creates or updates the NMT conda environment on group storage, installs the
# rational_cnn package from the repository root, i.e. the parent of nmt/
# (editable, so NMT and the DGMC experiments always agree on the basis
# implementations), pre-fetches the VGG16
# weights so that concurrent jobs do not race on the download, and runs the
# unit suite. Does not submit any Slurm job. Run on a login node.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "$NMT_REPO_ROOT"

RBCNN_REPO="${RBCNN_REPO:-$(cd "$NMT_REPO_ROOT/.." && pwd)}"
[[ -f "$RBCNN_REPO/setup.py" ]] || { echo "ERROR: the rational_cnn package (setup.py) not found at $RBCNN_REPO" >&2; exit 1; }

mkdir -p \
  "$NMT_RUN_ROOT" "$NMT_CACHE_ROOT" "$NMT_WANDB_ROOT" "$NMT_PFS_ROOT/tmp" \
  "$NMT_SLURM_LOG_ROOT" "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS"

module load lang/Miniforge3/25.3.0-3
if [[ -x "$NMT_CONDA_ENV/bin/python" ]]; then
  conda env update --prefix "$NMT_CONDA_ENV" --file environment-hpc.yml --prune -y
else
  mamba env create --prefix "$NMT_CONDA_ENV" --file environment-hpc.yml -y
fi

# The rational (and pure-PyTorch B-spline) convolutions live in that project.
"$NMT_CONDA_ENV/bin/python" -m pip install --no-deps -e "$RBCNN_REPO"

# Populate $TORCH_HOME now: 65 jobs starting at once should not each download.
"$NMT_CONDA_ENV/bin/python" - <<'PY'
import torchvision.models as models
models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
print('VGG16 ImageNet weights cached')
PY

# Tolerate "no tests collected" (exit 5) so a fresh checkout can be set up.
set +e
"$NMT_CONDA_ENV/bin/python" -m pytest -q
code=$?
set -e
[[ $code -eq 0 || $code -eq 5 ]] || exit $code
