#!/usr/bin/env bash
# Moves a finished dataset from the writable scratch staging area to the
# canonical dataset root on group storage. RUN ON A LOGIN NODE.
#
# WHY THIS EXISTS. Preparing a dataset needs two things that no single
# filesystem here provides at once:
#   * a GPU (PascalVOC runs VGG16 over every image), which means a compute node;
#   * write access to the canonical dataset root, which compute nodes do not
#     have because /pc2/groups is mounted read-only there.
# So preparation writes to $RBCNN_DATA_STAGE on /scratch (writable everywhere),
# and this script promotes the result to $RBCNN_DATA_ROOT from a login node,
# where group storage is writable. Training jobs then only ever read.
#
# Usage: bash hpc/promote_dataset.sh PascalVOC [FAUST ...]
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: run this on a login node; \$RBCNN_DATA_ROOT is read-only inside Slurm jobs." >&2
  exit 1
fi
if [[ ! -w "$RBCNN_DATA_ROOT" ]]; then
  echo "ERROR: $RBCNN_DATA_ROOT is not writable from here." >&2
  exit 1
fi
[[ $# -ge 1 ]] || { echo "usage: $0 <dataset-name> [...]" >&2; exit 1; }

for NAME in "$@"; do
  SRC="${RBCNN_DATA_STAGE}/${NAME}"
  DST="${RBCNN_DATA_ROOT}/${NAME}"
  [[ -d "$SRC" ]] || { echo "ERROR: no staged dataset at $SRC" >&2; exit 1; }

  if [[ -e "$DST" ]]; then
    echo "ERROR: $DST already exists. Remove it deliberately before re-promoting." >&2
    exit 1
  fi

  n_src=$(find "$SRC" | wc -l)
  echo "promoting ${NAME}: ${n_src} entries, $(du -sh "$SRC" | cut -f1)"
  cp -a "$SRC" "$DST"

  n_dst=$(find "$DST" | wc -l)
  [[ "$n_src" == "$n_dst" ]] || { echo "ERROR: entry count mismatch $n_src != $n_dst" >&2; exit 1; }

  echo "  verified ${n_dst} entries at $DST"
  echo "  staged copy left at $SRC -- delete it to reclaim /scratch inodes:"
  echo "      rm -rf '$SRC'"
done
