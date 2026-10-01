#!/bin/bash
# Trains the SALT MaskFormer variants one after another on the local GPU:
#   none      plain MaskFormer (Van Stroud et al. 2024)
#   spline    + SplineCNN pre-encoder (degree-1 B-splines, 3^4 = 81 hats)
#   rational  + rational-basis pre-encoder (K = 9, PCA-of-hats init)
# Usage: vertexing/run_maskformer.sh [seeds...]   (default: 1 2)
# Run from the repository root; needs the salt venv (see README.md).
set -u
export PYTHONPATH=vertexing${PYTHONPATH:+:$PYTHONPATH}  # maskformer_geo.py
SALT=${SALT:-$HOME/.venvs/salt/bin/salt}
OUT=${OUT:-vertexing/runs/main}
CFG=vertexing/configs/maskformer_geo.yaml
SEEDS=${@:-1 2}
mkdir -p "$OUT/logs"
for seed in $SEEDS; do
  for conv in none spline rational; do
    log="$OUT/logs/${conv}_s${seed}.log"
    if grep -q "max_epochs=.* reached" "$log" 2>/dev/null; then
      echo "skip $conv seed $seed (done)"; continue
    fi
    echo "$(date) start $conv seed $seed"
    "$SALT" fit --config "$CFG" \
      --model.model.init_args.encoder.init_args.conv=$conv \
      --seed_everything $seed \
      --data.num_workers 6 \
      --trainer.default_root_dir "$OUT/${conv}_s${seed}" \
      --force > "$log" 2>&1
    echo "$(date) end $conv seed $seed (exit $?)"
  done
done
