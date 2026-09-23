#!/usr/bin/env bash
# PC2 Noctua 2 environment for RationalBasisCNN.
# Source this file; do not execute it. It deliberately avoids writing to HOME:
# /pc2/users (HOME) is quota-limited, so datasets, downloads, caches and the
# conda environment all live on project storage.

export RBCNN_SLURM_ACCOUNT="hpc-prf-llmrout"
# Derived from this file location, so a second clone stays self-consistent.
export RBCNN_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Bulk, sequential data (datasets, caches, tmp) -> parallel filesystem.
export RBCNN_PFS_ROOT="/scratch/hpc-prf-llmrout/hpcabpo/rational-basis-cnn"
# Many small files (conda) -> group storage, as PC2 recommends for envs.
export RBCNN_GROUP_ROOT="/pc2/groups/hpc-prf-llmrout/hpcabpo/rational-basis-cnn"

# Datasets live on GROUP storage. /scratch allows only 1,000,000 inodes and raw
# dataset trees are mostly tiny files (PascalVOC 65k, SPair-71k 76k), which that
# quota cannot absorb.
#
# Group storage is mounted READ-ONLY on compute nodes. That is fine for training,
# because PyG skips both download and processing when the raw and processed files
# already exist, so a prepared dataset is only ever read. Anything that WRITES a
# dataset -- the prep jobs, which need a GPU and therefore a compute node -- uses
# $RBCNN_DATA_STAGE on /scratch, and hpc/promote_dataset.sh then moves the result
# to $RBCNN_DATA_ROOT from a login node, where group storage is writable.
export RBCNN_GROUP_DATA_ROOT="${RBCNN_GROUP_ROOT}/data"
export RBCNN_DATA_ROOT="${RBCNN_DATA_ROOT:-${RBCNN_GROUP_DATA_ROOT}}"   # --root for training (read-only on compute)
export RBCNN_DATA_STAGE="${RBCNN_DATA_STAGE:-${RBCNN_PFS_ROOT}/data}"    # writable everywhere, for prep jobs
export RBCNN_SPAIR_ROOT="${RBCNN_GROUP_DATA_ROOT}/SPair-71k"
export RBCNN_DOWNLOAD_ROOT="${RBCNN_PFS_ROOT}/downloads"
export RBCNN_CACHE_ROOT="${RBCNN_PFS_ROOT}/cache"
export RBCNN_WANDB_ROOT="${RBCNN_PFS_ROOT}/wandb"
export RBCNN_CONDA_ENV="${RBCNN_GROUP_ROOT}/conda/envs/rational-basis-cnn"

# Small text logs stay in the repository: slurm/logs/ is gitignored scratch,
# results/logs/ is what results/analyze.py reads and is tracked by git.
# Override RBCNN_RESULTS_LOG_ROOT to keep the paper's checked-in logs intact.
export RBCNN_SLURM_LOG_ROOT="${RBCNN_SLURM_LOG_ROOT:-${RBCNN_REPO_ROOT}/slurm/logs}"
export RBCNN_RESULTS_LOG_ROOT="${RBCNN_RESULTS_LOG_ROOT:-${RBCNN_REPO_ROOT}/results/logs}"

# Conda itself and every regenerable cache stay outside the HOME quota.
export CONDA_ENVS_PATH="${RBCNN_GROUP_ROOT}/conda/envs"
export CONDA_PKGS_DIRS="${RBCNN_GROUP_ROOT}/conda/pkgs"
export XDG_CACHE_HOME="${RBCNN_CACHE_ROOT}/xdg"
export TORCH_HOME="${RBCNN_CACHE_ROOT}/torch"        # torchvision VGG16 weights
export PIP_CACHE_DIR="${RBCNN_CACHE_ROOT}/pip"
export MPLCONFIGDIR="${RBCNN_CACHE_ROOT}/matplotlib"
export PYTHONPYCACHEPREFIX="${RBCNN_CACHE_ROOT}/pycache"
export CUDA_CACHE_PATH="${RBCNN_CACHE_ROOT}/nv"      # else ~/.nv/ComputeCache
export TRITON_CACHE_DIR="${RBCNN_CACHE_ROOT}/triton"  # else ~/.triton
export TORCHINDUCTOR_CACHE_DIR="${RBCNN_CACHE_ROOT}/inductor"
export RATIONAL_KERNEL_CACHE="${RBCNN_CACHE_ROOT}/rational-kernels"  # else ~/.cache

# Which implementation rational_cnn uses to evaluate the multivariate basis.
# Upstream commit 1d73538 makes the fused Triton kernels the default. They are
# mathematically equivalent to the eager path but associate the polynomial sums
# differently (nested Horner instead of one matmul), so results differ at ~1e-6
# and SGD amplifies that: upstream's own same-seed check came out 47.85 vs
# 48.48. Every PascalVOC, FAUST, SPair-71k and NMT number we have logged was
# produced by the eager path, so it is pinned here to keep re-runs comparable
# with the existing tables. Unset it for a fresh experiment whose whole table is
# Triton-produced -- but never mix the two within one table.
# Harmless before the upstream pull: nothing reads this variable yet.
export RATIONAL_BASIS_IMPL="${RATIONAL_BASIS_IMPL:-eager}"
export WANDB_DIR="${RBCNN_WANDB_ROOT}"
export WANDB_CACHE_DIR="${RBCNN_CACHE_ROOT}/wandb"
export WANDB_ARTIFACT_DIR="${RBCNN_CACHE_ROOT}/wandb-artifacts"
export PYTHONNOUSERSITE=1

# Put the project interpreter first, so `python ...` works after sourcing this
# file. The Slurm scripts still call $RBCNN_CONDA_ENV/bin/python explicitly, so
# they do not depend on PATH order.
if [[ -d "${RBCNN_CONDA_ENV}/bin" ]]; then
  export PATH="${RBCNN_CONDA_ENV}/bin:${PATH}"
fi

# On login nodes setup/download commands use this directory. Slurm jobs keep
# their isolated node-local /tmp allocation.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  export TMPDIR="${RBCNN_PFS_ROOT}/tmp"
fi
