#!/bin/bash
#SBATCH --job-name=nmt-voc
#SBATCH --account=hpc-prf-llmrout
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# One NMT run on PascalVOC-Keypoints on a single H100, for the experiment json
# in $CONFIG. Env: CONFIG (required), PYTHON (project env),
# WANDB (offline | online | disabled; default offline).
# Submit from the repo root, e.g.
#   CONFIG=experiments/voc_vgg16.json sbatch slurm/nmt_voc.sbatch
# or, for the whole sweep, DATASET=voc slurm/submit.sh
#
# 16 CPUs because data/data_loader_multigraph.py hardcodes num_workers=10.
# Wall time: cfg.TRAIN.EPOCH_ITERS is 2000 for both datasets, so an epoch costs
# about the same here as on SPair-71k (~2.2 h measured) and the 7-epoch budget
# lands near 15.3 h. 18 h leaves margin. PascalVOC images are smaller than
# SPair's, so if the first runs come in well under that, lower this limit --
# short jobs backfill far better on this partition.
# A run cut short still leaves the last epoch's checkpoint and its per-epoch
# evaluation, since both are written every epoch.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR}"
source "$ROOT/hpc/env.sh"
: "${CONFIG:?set CONFIG=experiments/<name>.json}"
cd "$ROOT"
[ -r "$CONFIG" ] || { echo "ERROR: no such config: $CONFIG" >&2; exit 1; }

# The inputs data/pascal_voc.py needs, checked before burning an
# allocation. All of them live in the PascalVOC tree shared with the DGMC
# experiments at the repository root; see NMT_VOC_ROOT in hpc/env.sh.
for p in "$NMT_VOC_ROOT/images/JPEGImages" "$NMT_VOC_ROOT/images/Annotations" \
         "$NMT_VOC_ROOT/annotations" "$NMT_VOC_ROOT/splits.npz"; do
  if [ ! -e "$p" ]; then
    echo "ERROR: PascalVOC input missing: $p" >&2
    echo "  the tree is shared with the DGMC experiments; see" >&2
    echo "  ../experiments/prepare_pascal_voc.py and ../hpc/promote_dataset.sh" >&2
    exit 1
  fi
done
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
print('dataset=%s backbone=%s image_size=%s conv=%s model_dir=%s' % (
    cfg.get('DATASET_NAME', '(inherited)'),
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
