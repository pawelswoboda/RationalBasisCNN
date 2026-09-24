#!/usr/bin/env bash
# Submits one NMT job per (configuration, seed) on a single H100 each -- the
# same configurations and seeds as the DGMC sweep `../slurm/submit.sh spair`
# at the repository root, so the two models can be compared row by row.
#
# Usage: slurm/submit.sh [config names...]     (default: the 13-config sweep)
# Env:   DATASET=spair|voc selects the benchmark (default spair)
#        SEEDS="0 1 2" overrides the seeds (default 0-4)
#        DRY_RUN=1 writes the configs and prints the jobs, submits nothing
#        WANDB, PYTHON are passed through to the jobs
#        slurm/make_config.py --list shows every available name
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
mkdir -p slurm/logs

# Pull just the variables we need, so env.sh's login-node TMPDIR is not
# exported into the jobs.
eval "$(source hpc/env.sh >/dev/null && printf 'RUN_ROOT=%q\nPYBIN=%q\nSPAIR=%q\nVOC=%q\nENVDIR=%q\n' \
        "$NMT_RUN_ROOT" "$NMT_CONDA_ENV/bin/python" "$NMT_SPAIR_ROOT" "$NMT_VOC_ROOT" "$NMT_CONDA_ENV")"

# Which benchmark to sweep. Each entry fixes the dataset name passed to
# make_config.py, the sbatch script, the run subdirectory and the one path whose
# absence means the data was never staged.
DATASET="${DATASET:-spair}"
case "$DATASET" in
  spair) SBATCH=slurm/nmt_spair.sbatch; RUN_SUBDIR=spair_vgg16; DATA_PROBE="$SPAIR/PairAnnotation" ;;
  voc)   SBATCH=slurm/nmt_voc.sbatch;   RUN_SUBDIR=voc_vgg16;   DATA_PROBE="$VOC/splits.npz" ;;
  *)     echo "ERROR: DATASET must be 'spair' or 'voc', got '$DATASET'" >&2; exit 1 ;;
esac

if [ ! -e "$DATA_PROBE" ] || [ ! -x "$PYBIN" ]; then
  echo "ERROR: the $DATASET data ($DATA_PROBE) or the conda env is missing." >&2
  echo "       bash hpc/create_environment.sh   (login node)" >&2
  exit 1
fi

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  mapfile -t CONFIGS < <("$PYBIN" -c "import sys; sys.path.insert(0, 'slurm'); import make_config; print('\n'.join(make_config.DEFAULT_SWEEP))")
fi
SEEDS=(${SEEDS:-0 1 2 3 4})

GEN_DIR="$RUN_ROOT/configs"
mkdir -p "$GEN_DIR"

# Fail before submitting anything if a name is unknown.
for c in "${CONFIGS[@]}"; do
  "$PYBIN" -c "
import sys; sys.path.insert(0, 'slurm'); import make_config
raise SystemExit(0 if '$c' in make_config.CONFIGS else 'unknown configuration: $c')"
done

echo "$DATASET: ${#CONFIGS[@]} configurations x ${#SEEDS[@]} seeds = $(( ${#CONFIGS[@]} * ${#SEEDS[@]} )) jobs"
for c in "${CONFIGS[@]}"; do
  for s in "${SEEDS[@]}"; do
    out="$GEN_DIR/${DATASET}_${c}_seed${s}.json"
    "$PYBIN" slurm/make_config.py "$c" "$s" "$out" "$RUN_ROOT/$RUN_SUBDIR" "$DATASET" >/dev/null
    if [ -n "${DRY_RUN:-}" ]; then
      echo "[dry run] CONFIG=$out sbatch --job-name=nmt-$DATASET-$c-s$s $SBATCH"
      continue
    fi
    id=$(CONFIG="$out" sbatch --parsable --job-name="nmt-$DATASET-$c-s$s" "$SBATCH")
    echo "$c seed $s -> job $id"
  done
done
