#!/usr/bin/env bash
# Downloads HPatches (the 116 image sequences, not the patch archive) for the
# out-of-task transfer experiment. Idempotent: re-running verifies instead of
# re-downloading. Run on a login node.
#
# PROVENANCE. The original host icvl.ee.ic.ac.uk is unreachable; this uses the
# dataset author's HuggingFace mirror, which the official hpatches-dataset
# README links to.
#
# LAYOUT. Extracted to group storage next to the other datasets: 1463 files,
# which the /scratch inode quota (1,000,000, ~89% used) cannot spare.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

URL="https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip"
ZIP="${RBCNN_DOWNLOAD_ROOT}/hpatches-sequences-release.zip"
SHA256="99cf7e1ca167896eb4ca2fe3d903beff6566243a3c0c8e07a13a2596648ad670"
SIZE=1279545572
TARGET="${RBCNN_DATA_ROOT}/HPatches"

mkdir -p "$RBCNN_DOWNLOAD_ROOT" "$RBCNN_DATA_ROOT"
if [[ ! -s "$ZIP" ]]; then
  echo "downloading HPatches sequences (~1.2 GiB)"
  curl -L --fail --retry 3 --retry-delay 5 -C - -o "$ZIP" "$URL"
fi
actual=$(stat -c %s "$ZIP")
[[ "$actual" == "$SIZE" ]] || { echo "size mismatch: $actual != $SIZE" >&2; exit 1; }
echo "${SHA256}  ${ZIP}" | sha256sum -c -

if [[ ! -d "$TARGET" ]]; then
  echo "extracting to $TARGET"
  tmp="${RBCNN_DATA_ROOT}/.hpatches_tmp"
  rm -rf "$tmp"; mkdir -p "$tmp"
  "${RBCNN_CONDA_ENV}/bin/python" -c "
import zipfile; zipfile.ZipFile('$ZIP').extractall('$tmp')"
  mv "$tmp/hpatches-sequences-release" "$TARGET"
  rmdir "$tmp"
fi

n=$(ls "$TARGET" | wc -l)
[[ "$n" == "116" ]] || { echo "expected 116 sequences, found $n" >&2; exit 1; }
echo "HPatches ready: $n sequences ($(ls "$TARGET" | grep -c '^v_') viewpoint, $(ls "$TARGET" | grep -c '^i_') illumination) at $TARGET"
