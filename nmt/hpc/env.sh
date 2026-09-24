#!/usr/bin/env bash
# PC2 Noctua 2 environment for NormMatchTrans (NMT).
# Source this file; do not execute it. $HOME is quota-limited, so the conda
# environment, datasets, run outputs and every cache live on project storage.
# See docs/hpc_runbook.md at the repository root for the storage conventions.

export NMT_SLURM_ACCOUNT="hpc-prf-llmrout"
# Derived from this file location, so a second clone stays self-consistent.
export NMT_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Writable everywhere (GPFS): run outputs, checkpoints, caches, tmp.
export NMT_PFS_ROOT="/scratch/hpc-prf-llmrout/hpcabpo/nmt"
# Read-only on compute nodes (NFS): the conda environment.
export NMT_GROUP_ROOT="/pc2/groups/hpc-prf-llmrout/hpcabpo/nmt"
export NMT_CONDA_ENV="${NMT_GROUP_ROOT}/conda/envs/nmt"

# The raw SPair-71k tree is shared with the DGMC experiments instead of
# downloaded twice; NMT reads the images and annotations directly (it does no
# feature pre-computation), and reading group storage works inside jobs.
export NMT_SPAIR_ROOT="/pc2/groups/hpc-prf-llmrout/hpcabpo/rational-basis-cnn/data/SPair-71k"
# PascalVOC-Keypoints is shared the same way. NMT needs three things from that
# tree and all three are already present, so nothing is downloaded twice:
#   $NMT_VOC_ROOT/images/       VOC2011, holding JPEGImages/ and Annotations/
#   $NMT_VOC_ROOT/annotations/  Berkeley keypoint annotations, one dir per class
#   $NMT_VOC_ROOT/splits.npz    the train/test split. data/pascal_voc.py reads
#       cfg.VOC2011.SET_SPLIT as f[sets] indexed by class, which is exactly this
#       file's layout (an object array of 20 per-class xml lists, in the order of
#       cfg.VOC2011.CLASSES), so ThinkMatch's voc2011_pairs.npz is not needed.
export NMT_VOC_ROOT="/pc2/groups/hpc-prf-llmrout/hpcabpo/rational-basis-cnn/data/PascalVOC/raw"
# Checkpoints and logs are WRITTEN by jobs, so they must be on /scratch.
export NMT_RUN_ROOT="${NMT_PFS_ROOT}/runs"
export NMT_CACHE_ROOT="${NMT_PFS_ROOT}/cache"
export NMT_WANDB_ROOT="${NMT_PFS_ROOT}/wandb"
export NMT_SLURM_LOG_ROOT="${NMT_SLURM_LOG_ROOT:-${NMT_REPO_ROOT}/slurm/logs}"

# Conda itself and every regenerable cache stay outside the HOME quota.
export CONDA_ENVS_PATH="${NMT_GROUP_ROOT}/conda/envs"
export CONDA_PKGS_DIRS="${NMT_GROUP_ROOT}/conda/pkgs"
export XDG_CACHE_HOME="${NMT_CACHE_ROOT}/xdg"
export TORCH_HOME="${NMT_CACHE_ROOT}/torch"        # pretrained VGG16 weights
export PIP_CACHE_DIR="${NMT_CACHE_ROOT}/pip"
export MPLCONFIGDIR="${NMT_CACHE_ROOT}/matplotlib"
export PYTHONPYCACHEPREFIX="${NMT_CACHE_ROOT}/pycache"
export CUDA_CACHE_PATH="${NMT_CACHE_ROOT}/nv"      # else ~/.nv/ComputeCache
export TRITON_CACHE_DIR="${NMT_CACHE_ROOT}/triton"  # else ~/.triton
export TORCHINDUCTOR_CACHE_DIR="${NMT_CACHE_ROOT}/inductor"
export RATIONAL_KERNEL_CACHE="${NMT_CACHE_ROOT}/rational-kernels"  # else ~/.cache

# hpc/create_environment.sh installs the rational_cnn package of the repository
# root EDITABLE, so NMT always runs the repository's basis code. Commit 1d73538
# made fused Triton kernels the default (rational_cnn/triton_basis.py reads this
# variable); they are mathematically equivalent to the eager path but associate
# the polynomial sums differently, so results move by more than the margins
# these sweeps measure. Every NMT run in results/logs/nmt_spair and nmt_voc used
# eager (the first SPair-71k sweep predates the Triton kernels), so it is pinned
# here to keep new runs comparable with them.
export RATIONAL_BASIS_IMPL="${RATIONAL_BASIS_IMPL:-eager}"
export WANDB_DIR="${NMT_WANDB_ROOT}"
export WANDB_CACHE_DIR="${NMT_CACHE_ROOT}/wandb"
export WANDB_ARTIFACT_DIR="${NMT_CACHE_ROOT}/wandb-artifacts"
export PYTHONNOUSERSITE=1

# Put the project interpreter first, so `python ...` works after sourcing this
# file. The Slurm scripts still call $NMT_CONDA_ENV/bin/python explicitly, so
# they do not depend on PATH order.
if [[ -d "${NMT_CONDA_ENV}/bin" ]]; then
  export PATH="${NMT_CONDA_ENV}/bin:${PATH}"
fi

# On login nodes setup commands use this directory. Slurm jobs keep their
# isolated node-local /tmp allocation.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  export TMPDIR="${NMT_PFS_ROOT}/tmp"
fi
