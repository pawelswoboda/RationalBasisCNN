#!/usr/bin/env bash
# Submits one slurm job per (config, seed).
# Usage: slurm/submit.sh voc   [configs...]   (default: the PascalVOC configs of the paper, seeds 0-4)
#        slurm/submit.sh faust [configs...]   (default: the FAUST configs of the paper, seeds 0-2)
# Env: SEEDS="0 1 2" overrides the seed list; PYTHON, WANDB, EPOCHS are passed through.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
WHAT="${1:?usage: slurm/submit.sh voc|faust [configs...]}"; shift || true
mkdir -p slurm/logs
case "$WHAT" in
  voc)
    CONFIGS=("$@"); [ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(spline spline_k3 spline_k2 rational rational_k3 rational_mv_k3 mv_K4 mv_K6 mv_K9 mv_K16 mlp_K4 mlp_K9 rational_mv_d54)
    SEEDS=(${SEEDS:-0 1 2 3 4}); SBATCH=slurm/pascal_voc.sbatch ;;
  faust)
    CONFIGS=("$@"); [ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(faust_mv_K8 faust_mv_K4 faust_rational_k3 faust_mv_K4_mean faust_spline_mean faust_mlp_K8 faust_spline faust_spline_k3 faust_mv_K16 faust_mv_K27 faust_spline_noclip faust_spline_k2)
    SEEDS=(${SEEDS:-0 1 2}); SBATCH=slurm/faust.sbatch ;;
  *) echo "unknown target '$WHAT'" >&2; exit 1 ;;
esac
source slurm/configs.sh
for c in "${CONFIGS[@]}"; do
  config_args "$c" >/dev/null   # fail early on unknown config
  for s in "${SEEDS[@]}"; do
    id=$(CONFIG="$c" SEED="$s" sbatch --parsable --job-name="$WHAT-$c-s$s" "$SBATCH")
    echo "$WHAT $c seed $s -> job $id"
  done
done
