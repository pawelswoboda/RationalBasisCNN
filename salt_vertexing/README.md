# SplineCNN / rational-basis GNN in front of MaskFormer secondary-vertex reconstruction

Experiment 6: does a continuous-kernel graph convolution over track geometry
help the MaskFormer secondary-vertex finder of

> S. Van Stroud, P. Duckett, M. Hart, N. Pond, S. Rettie, G. Facini, T. Scanlon,
> *Secondary vertex reconstruction with MaskFormers*, EPJC 84 (2024) 1020,
> [arXiv:2312.12272](https://arxiv.org/abs/2312.12272)?

The host framework is ATLAS's **SALT**
([gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt),
docs at [ftag-salt.docs.cern.ch](https://ftag-salt.docs.cern.ch)), used
unmodified. The data is the paper's public Delphes ttbar sample
([zenodo.org/records/10371998](https://zenodo.org/records/10371998)): jets with
up to 50 tracks and up to 5 truth b/c hadrons, each track linked to its hadron.

## What is prepended

`rational_cnn/salt_encoder.py:GeometricPreEncoder` wraps SALT's Transformer
encoder. Per jet it

1. builds a kNN graph (k = 8) over the tracks in (η_rel, φ_rel);
2. takes pairwise differences of (η_rel, φ_rel, d0, z0) (normalised units) as
   D = 4 pseudo-coordinates, squashed to [0, 1] by
   `u = ½ + ½ tanh(asinh(Δ/s)/2)` with s = (1, 1, 0.01, 0.01) — the asinh
   handles the heavy tails of impact-parameter differences (in-jet median
   |Δd0| ≈ 0.007, 95th percentile ≈ 1);
3. applies 2 residual pre-norm graph convolutions of width 64:
   `spline` (`BSplineConv`, degree-1 hats, 3⁴ = 81 bases),
   `rational` (`RationalConv`, multivariate safe-Padé basis, K = 9,
   degrees (4, 3), PCA-of-hats init), or `mlp` (MLP basis, K = 9, control);
4. adds the result to the track embeddings through a **zero-initialised**
   projection, so every variant starts as exactly the plain MaskFormer.

The GNN runs in fp32 inside SALT's fp16 autocast. Everything after it (4-layer
Transformer, 5-query 3-layer MaskFormer decoder, jet-flavour / track-origin /
vertex-regression heads, losses and Hungarian matching) is SALT's MaskFormer
recipe (`salt/configs/MaskFormer.yaml`), adapted to the variables of the public
dataset (`configs/maskformer_geo.yaml`).

## Setup

```bash
# SALT (public clone) into its own venv; torch is pinned by salt
git clone https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt.git ../external/salt
uv venv -p 3.11 ~/.venvs/salt
VIRTUAL_ENV=~/.venvs/salt uv pip install -e ../external/salt torch_geometric seaborn pytest
VIRTUAL_ENV=~/.venvs/salt uv pip install --no-deps -e .          # rational_cnn
# CPUs without AVX (e.g. the local i7-860): the py-lap-solver wheel is built
# with -march=native and dies with "Illegal instruction"; build it from the
# sdist (fix its pyproject: cmake.minimum-version -> cmake.version = ">=3.24")
#   VIRTUAL_ENV=~/.venvs/salt uv pip install --reinstall ./py_lap_solver-0.1.4

# data (15 GB) into data/vertexing/
for f in norm_dict.yaml class_dict.yaml pp_output_val.h5 pp_output_test_ttbar.h5 pp_output_train.h5; do
  curl -L -o data/vertexing/$f https://zenodo.org/api/records/10371998/files/$f/content
done
```

## Running

```bash
~/.venvs/salt/bin/python -m pytest salt_vertexing/tests     # 5 tests
salt_vertexing/run_all.sh 1 2                              # 4 variants x seeds 1, 2
~/.venvs/salt/bin/python salt_vertexing/summarize.py salt_vertexing/runs/main
```

One run: `salt fit --config salt_vertexing/configs/maskformer_geo.yaml
--model.model.init_args.encoder.init_args.conv={none,spline,rational,mlp} --force`.
Budget: 2 M training jets × 15 epochs, batch 1000, OneCycle (peak 5e-4) —
about 1.6 h per run on an RTX 4070 Ti Super, ~1/20 of the paper's compute
(13.5 M jets × 50 epochs), so absolute numbers are below the paper's.
Metrics (`metrics.csv` per run, from SALT's `MaskformerMetrics`) are on 200 k
validation jets at the epoch with the lowest validation loss:
perfect-match (recall = purity = 100 %) and loose-match (≥ 50 % / ≥ 50 %)
vertex efficiency and fake rate, per-class b/c vertex efficiency, jet-flavour
cross entropy.

## Results

See `results.md` (filled in when the runs finish).
