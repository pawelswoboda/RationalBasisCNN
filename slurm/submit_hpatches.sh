#!/usr/bin/env bash
# One HPatches transfer job per NMT configuration (all its seeds inside the job).
# Usage: slurm/submit_hpatches.sh [config names...]
#        (default: the four that carry the comparison)
# Env: SEEDS, NUM_KEYPOINTS, DISTRACTORS pass through; DRY_RUN=1 submits nothing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
mkdir -p slurm/logs

CONFIGS=("$@")
# spline is the fixed-basis baseline, rational_mv_k3 and mv_K4 the learnable
# ones at 2.6x and 5x fewer parameters, mlp_K4 the non-rational control.
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(spline rational_mv_k3 mv_K4 mlp_K4)

eval "$(source hpc/env.sh >/dev/null && printf 'DATA=%q\n' "$RBCNN_DATA_ROOT")"
[ -d "$DATA/HPatches" ] || { echo "ERROR: HPatches missing; run hpc/download_hpatches.sh" >&2; exit 1; }

for c in "${CONFIGS[@]}"; do
  if [ -n "${DRY_RUN:-}" ]; then
    echo "[dry run] CONFIG=$c sbatch --job-name=hpatches-$c slurm/hpatches.sbatch"
    continue
  fi
  id=$(CONFIG="$c" sbatch --parsable --job-name="hpatches-$c" slurm/hpatches.sbatch)
  echo "$c -> job $id"
done
