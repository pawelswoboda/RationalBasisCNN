#!/bin/bash
#SBATCH --job-name=nmt
#SBATCH --account=hpc-prf-llmrout
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# One NMT run on SPair-71k on a single H100, for the experiment json in
# $CONFIG. Env: CONFIG (required), PYTHON (project env),
# WANDB (offline | online | disabled; default offline).
# Submit from the repo root, e.g.
#   CONFIG=experiments/spair_vgg16.json sbatch slurm/nmt_spair.sbatch
#
# 16 CPUs because data/data_loader_multigraph.py hardcodes num_workers=10.
# Wall time: NMT trains about 11 epochs per day, i.e. ~2.2 h per epoch, and the
# configs stop after 7 epochs (cfg.TRAIN.max_epochs), so ~15.3 h plus margin.
# 18 h covers ~8.3 epochs at that rate.
# A run cut short by the limit still leaves the last epoch's checkpoint and its
# per-epoch evaluation, since both are written every epoch. That per-epoch rate
# was measured with the SwinV2-L backbone; VGG16 is far smaller, so if the
# first runs are quicker, lower this -- short limits backfill much better.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR}"
source "$ROOT/hpc/env.sh"
: "${CONFIG:?set CONFIG=experiments/<name>.json}"
cd "$ROOT"
[ -r "$CONFIG" ] || { echo "ERROR: no such config: $CONFIG" >&2; exit 1; }

if [ ! -d "$NMT_SPAIR_ROOT/PairAnnotation" ]; then
  echo "ERROR: SPair-71k not found at $NMT_SPAIR_ROOT" >&2
  echo "  it is shared with the DGMC experiments; see ../hpc/download_spair71k.sh" >&2
  exit 1
fi
if [ ! -x "$NMT_CONDA_ENV/bin/python" ]; then
  echo "ERROR: conda env missing; run 'bash hpc/create_environment.sh' on a login node" >&2
  exit 1
fi

PYTHON="${PYTHON:-$NMT_CONDA_ENV/bin/python}"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 10 data-loader workers each spawning a full thread pool would oversubscribe
# the 16 allocated cores.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
# train_eval.py calls wandb.init() unconditionally; offline keeps the run data
# on scratch without needing credentials in the job (sync it later with
# `wandb sync`). Set WANDB=online once you have logged in, or =disabled.
export WANDB_MODE="${WANDB:-offline}"
export WANDB_DIR="$NMT_WANDB_ROOT"
mkdir -p "$NMT_WANDB_ROOT" "$NMT_RUN_ROOT"

echo "job=${SLURM_JOB_ID} node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) config=$CONFIG wandb=$WANDB_MODE"
$PYTHON - "$CONFIG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
print('backbone=%s image_size=%s conv=%s model_dir=%s' % (
    cfg.get('BACKBONE', '(inherited)'), cfg.get('IMAGE_SIZE', '(inherited)'),
    cfg.get('SPLINE_CNN', {}).get('conv', '(inherited)'), cfg.get('model_dir')))
PY

# One process, one GPU: train_eval.py hardcodes world_size=1 / local_rank=0.
# The rendezvous port must be free on this node. Deriving it from the job id
# collides whenever two co-scheduled jobs' ids differ by a multiple of the
# modulus -- that cost one run of the first PascalVOC sweep with EADDRINUSE
# seven seconds in -- so ask the kernel for a free port instead and re-check it
# is still free just before binding.
for attempt in 1 2 3 4 5; do
  PORT=$($PYTHON -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
  $PYTHON -c "
import socket, sys
s = socket.socket()
try:
    s.bind(('', $PORT))
except OSError:
    sys.exit(1)
finally:
    s.close()" && break
  echo "port $PORT taken, retrying ($attempt/5)" >&2
done
echo "rendezvous port=$PORT"
$PYTHON -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="$PORT" \
  train_eval.py "$CONFIG"
