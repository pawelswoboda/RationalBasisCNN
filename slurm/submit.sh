#!/usr/bin/env bash
# Submits one slurm job per (config, seed).
# Usage: slurm/submit.sh voc   [configs...]   (default: the PascalVOC configs of the paper, seeds 0-4)
#        slurm/submit.sh faust [configs...]   (default: the FAUST configs of the paper, seeds 0-2)
#        slurm/submit.sh spair [configs...]   (default: the PascalVOC paper configs on SPair-71k, seeds 0-4)
# Env: SEEDS="0 1 2" overrides the seed list; PYTHON, WANDB, EPOCHS (and for
# spair LAYOUT, NUM_WORKERS) are passed through. DRY_RUN=1 prints the jobs
# without submitting anything.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
WHAT="${1:?usage: slurm/submit.sh voc|faust|spair [configs...]}"; shift || true
mkdir -p slurm/logs
case "$WHAT" in
  voc)
    CONFIGS=("$@"); [ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(spline spline_k3 spline_k2 rational rational_k3 rational_mv_k3 mv_K4 mv_K6 mv_K9 mv_K16 mlp_K4 mlp_K9 rational_mv_d54)
    SEEDS=(${SEEDS:-0 1 2 3 4}); SBATCH=slurm/pascal_voc.sbatch ;;
  faust)
    CONFIGS=("$@"); [ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(faust_mv_K8 faust_mv_K4 faust_rational_k3 faust_mv_K4_mean faust_spline_mean faust_mlp_K8 faust_spline faust_spline_k3 faust_mv_K16 faust_mv_K27 faust_spline_noclip faust_spline_k2)
    SEEDS=(${SEEDS:-0 1 2}); SBATCH=slurm/faust.sbatch ;;
  spair)
    CONFIGS=("$@"); [ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(spline spline_k3 spline_k2 rational rational_k3 rational_mv_k3 mv_K4 mv_K6 mv_K9 mv_K16 mlp_K4 mlp_K9 rational_mv_d54)
    SEEDS=(${SEEDS:-0 1 2 3 4}); SBATCH=slurm/spair71k.sbatch
    # Every job would fail at its guard if the graphs were never promoted, so
    # refuse up front. The subshell keeps env.sh's login-node TMPDIR out of the
    # environment the jobs inherit.
    if ! ( source hpc/env.sh && [ -r "$RBCNN_DATA_ROOT/SPair71k/processed/pairs.pt" ] ); then
      echo "ERROR: SPair71k is not prepared; run 'sbatch slurm/prep_spair71k.sbatch'," >&2
      echo "       then 'bash hpc/promote_dataset.sh SPair71k' on a login node." >&2
      exit 1
    fi ;;
  *) echo "unknown target '$WHAT'" >&2; exit 1 ;;
esac
source slurm/configs.sh
for c in "${CONFIGS[@]}"; do
  config_args "$c" >/dev/null   # fail early on unknown config
  for s in "${SEEDS[@]}"; do
    if [ -n "${DRY_RUN:-}" ]; then
      echo "[dry run] CONFIG=$c SEED=$s sbatch --job-name=$WHAT-$c-s$s $SBATCH"
      continue
    fi
    id=$(CONFIG="$c" SEED="$s" sbatch --parsable --job-name="$WHAT-$c-s$s" "$SBATCH")
    echo "$WHAT $c seed $s -> job $id"
  done
done
