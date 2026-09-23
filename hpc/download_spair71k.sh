#!/usr/bin/env bash
# Downloads and extracts SPair-71k (Min et al., ICCV 2019) with its keypoint
# annotations. Idempotent: re-running verifies rather than re-downloads.
#
# PROVENANCE. The official host cvlab.postech.ac.kr is unreachable (DNS
# resolves, but ports 80 and 443 both time out), so this fetches the Internet
# Archive's 2025-07-01 snapshot of the original tarball -- the same fallback
# the repository already uses for the PascalVOC keypoint annotations in
# experiments/prepare_pascal_voc.py. The snapshot's
# x-archive-orig-content-length matches its content-length, so it is the
# complete original file.
#
# LAYOUT. /scratch enforces a 1,000,000 inode quota and SPair-71k is 76,438
# mostly-tiny JSON files, so it is extracted to $RBCNN_DATA_ROOT on group
# storage, alongside PascalVOC. That root is read-only on compute nodes, which
# is fine: jobs only read the raw tree. The graphs built from it
# (slurm/prep_spair71k.sbatch) are staged on /scratch and promoted.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

URL="https://web.archive.org/web/20250701134122id_/https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz"
TARBALL="${RBCNN_DOWNLOAD_ROOT}/SPair-71k.tar.gz"
SHA256="d145eebe4d7d02ff0b9e2746dd3d72cfcb872df0dcfd6f3227576eb594408809"
SIZE=226961117
ENTRIES=76438

mkdir -p "$RBCNN_DOWNLOAD_ROOT" "$RBCNN_GROUP_DATA_ROOT" "$RBCNN_DATA_ROOT"

if [[ ! -s "$TARBALL" ]]; then
  echo "downloading SPair-71k (~217 MiB) from the Internet Archive snapshot"
  curl -L --fail --retry 3 --retry-delay 5 -C - -o "$TARBALL" "$URL"
fi

echo "verifying $TARBALL"
actual_size=$(stat -c %s "$TARBALL")
[[ "$actual_size" == "$SIZE" ]] || { echo "size mismatch: $actual_size != $SIZE" >&2; exit 1; }
echo "${SHA256}  ${TARBALL}" | sha256sum -c -
gzip -t "$TARBALL"

if [[ ! -d "$RBCNN_SPAIR_ROOT" ]]; then
  echo "extracting to $RBCNN_SPAIR_ROOT (76,438 files, a few minutes on group storage)"
  tar xzf "$TARBALL" -C "$RBCNN_GROUP_DATA_ROOT" --no-same-owner
fi

n=$(find "$RBCNN_SPAIR_ROOT" | wc -l)
[[ "$n" == "$ENTRIES" ]] || { echo "expected $ENTRIES entries, found $n" >&2; exit 1; }


echo "SPair-71k ready:"
echo "  categories : $(ls "$RBCNN_SPAIR_ROOT/ImageAnnotation" | wc -l)  (expected 18)"
echo "  images     : $(find "$RBCNN_SPAIR_ROOT/JPEGImages" -name '*.jpg' | wc -l)  (expected 1800)"
echo "  pairs      : $(find "$RBCNN_SPAIR_ROOT/PairAnnotation" -name '*.json' | wc -l)  (expected 70958)"
echo "  path       : $RBCNN_SPAIR_ROOT"
