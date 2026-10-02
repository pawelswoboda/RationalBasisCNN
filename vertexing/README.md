# Secondary-vertex reconstruction with rational-basis graph convolutions

Experiment 6: continuous-kernel graph convolutions with learnable rational
bases for reconstructing secondary vertices in jets, against the MaskFormer
of

> S. Van Stroud et al., *Secondary vertex reconstruction with MaskFormers*,
> EPJC 84 (2024) 1020, [arXiv:2312.12272](https://arxiv.org/abs/2312.12272).

Data: the paper's public Delphes ttbar sample
([zenodo.org/records/10371998](https://zenodo.org/records/10371998)), jets with
up to 50 tracks and up to 5 truth b/c hadrons, each track linked to its hadron
(`truth_hadron_idx`) and its production vertex (`truth_vertex_idx`), with an
8-class origin label.

## Approaches (one file each)

| file | approach |
|---|---|
| `rational_gnn_multicut.py` | **pure rational-basis GNN** + edge classification + multicut |
| `transformer_edge_multicut.py` | Transformer + MLP edge classifier + multicut (the paper's "EdgeClassifier" baseline) |
| `maskformer_geo.py` + `configs/maskformer_geo.yaml` | SALT MaskFormer, optionally with a SplineCNN / rational-basis pre-encoder |

### `rational_gnn_multicut.py`

* **Graph**: complete per jet (median 16, at most 50 tracks), dense
  `[B, L, L]` tensors.
* **Pseudo-coordinates** (D = 7) from the geometry of each track pair in the
  jet frame, tracks as straight lines in the transverse plane:
  sin/cos encoding of the polar-angle (from η) and azimuth differences —
  sin Δθ, 1 − cos Δθ, sin Δφ, 1 − cos Δφ —, the signed position `L_ij` along
  the jet axis of the crossing point of the two tracks (a 2-track vertex
  candidate), the z mismatch `dz_ij` of the two tracks at that crossing, Δz0.
  Squashed by `u = ½ + ½ tanh(asinh(Δ/s)/2)` (the non-negative 1 − cos terms
  by `u = tanh(asinh(Δ/s)/2)`, so that the small in-jet angles keep their
  resolution), s = (0.1, 0.005, 0.1, 0.005, 0.5 mm, 0.1 mm, 0.1 mm).
  On validation jets, same-hadron pairs cross a median 3.9 mm downstream
  (90 % quantile 83 mm), primary-vertex and pileup pairs at L ≈ 0.
* **Layers**: 4 × `h ← h + RationalConv(ReLU(LN(h)), u)`, width 128, K = 12
  multivariate safe-Padé bases (total degrees 4/3: 330 + 119 coefficients
  each), initialised to the 12 leading principal components of the 2⁷
  multilinear hats (fit on a 5⁷ grid, on the GPU: 14 s, > 20 min on the local
  CPU), mean aggregation; the free K is what makes D = 7 affordable (a
  B-spline grid would need ≥ 2⁷ = 128 hats per layer).
* **Edge classifier**, also a rational kernel:
  `logit_ij = Σ_p B_p(u_ij) [(U h_i)ᵀ diag(w_p) (V h_j) + β_p] + b`, rank 32,
  symmetrised. Plus a per-track origin head.
* **Loss**: BCE over track pairs (`--edge_label vertex`: same
  `truth_vertex_idx`, default; `hadron`: same `truth_hadron_idx`) + 0.5 ×
  origin cross entropy (class_dict weights capped at 50).
* 0.9 M parameters, ~8 k jets/s training on the RTX 4070 Ti Super.

### `transformer_edge_multicut.py`

Same data, losses, clustering and metrics; track features only (no pair
geometry), 4-layer pre-norm Transformer (d = 256, 8 heads, FF 512, as the
paper's encoder), attention-pooled jet vector, edge MLP on
`[h_i, h_j, g]` (128-64-32, first layer split per track), symmetrised.
2.4 M parameters, ~11 k jets/s.

### Clustering and metrics (both edge approaches)

* **Multicut** of the complete graph with costs `logit_ij − bias` (log-odds,
  positive = attractive), greedy additive edge contraction (GAEC) per jet;
  **connected components** of `{logit_ij > bias}` for comparison
  (`--solvers multicut cc`, `--biases ...` at eval).
* Clusters whose summed origin probability is mostly heavy flavour (fromBC +
  fromB + fromC) are the predicted secondary vertices.
* Scored like SALT's `MaskformerMetrics`: one-to-one (Hungarian, by shared
  tracks) matching of predicted vertices to truth hadrons with ≥ 1 track;
  efficiency = matched truth vertices passing recall/purity cuts / truth
  vertices, fake rate = predicted vertices not passing / predicted vertices;
  *perfect* (100 % / 100 %) and *loose* (≥ 50 % / ≥ 50 %), split by b and c.
* Caveat: ~10 % of fromB / fromBC / fromC tracks have no hadron link in the
  dataset, so no truth vertex contains them; a correct origin head calls them
  heavy flavour and their clusters count as fakes. MaskFormer's targets
  exclude them, so the fake rates are not yet like-for-like.

### `maskformer_geo.py` (SALT)

`GeometricPreEncoder` wraps SALT's Transformer: per-jet kNN graph (k = 8) in
(η, φ), pairwise (Δη, Δφ, Δd0, Δz0) pseudo-coordinates, 2 residual
`BSplineConv` (3⁴ = 81 hats) or `RationalConv` (K = 9) layers of width 64,
zero-initialised projection onto the track embeddings (`conv=none` is the
plain MaskFormer). Plus `MetricsCSV`, which writes SALT's validation metrics
to `metrics.csv`.

## Setup

```bash
git clone https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt.git ../external/salt
uv venv -p 3.11 ~/.venvs/salt
VIRTUAL_ENV=~/.venvs/salt uv pip install -e ../external/salt torch_geometric seaborn pytest
VIRTUAL_ENV=~/.venvs/salt uv pip install --no-deps -e .          # rational_cnn
# CPUs without AVX (e.g. the local i7-860): the py-lap-solver wheel is built
# with -march=native and dies with "Illegal instruction"; build it from the
# sdist (fix its pyproject: cmake.minimum-version -> cmake.version = ">=3.24")

mkdir -p data/vertexing   # 15 GB
for f in norm_dict.yaml class_dict.yaml pp_output_val.h5 pp_output_test_ttbar.h5 pp_output_train.h5; do
  curl -L -o data/vertexing/$f https://zenodo.org/api/records/10371998/files/$f/content
done
```

The edge approaches need only torch, h5py, scipy, pyyaml and `rational_cnn`
(no SALT).

## Running

```bash
PY=~/.venvs/salt/bin/python
$PY -m pytest vertexing/tests                                   # 11 tests

$PY vertexing/rational_gnn_multicut.py train --out vertexing/runs/rgnn_s1 --seed 1
$PY vertexing/transformer_edge_multicut.py train --out vertexing/runs/tedge_s1 --seed 1
$PY vertexing/rational_gnn_multicut.py eval --ckpt vertexing/runs/rgnn_s1/best.pt \
    --file data/vertexing/pp_output_test_ttbar.h5 --biases -1 0 1 --save rgnn_s1_test.json

vertexing/run_maskformer.sh 1 2                                 # none / spline / rational x seeds
$PY vertexing/summarize_maskformer.py vertexing/runs/main
```

Defaults: 2 M training jets × 15 epochs, batch 1000, AdamW, OneCycle
1e-7 → 5e-4 → 1e-5 (as SALT's MaskFormer config), validation on 200 k jets
(vertex metrics on the first 50 k each epoch), best epoch by validation loss.
Per-epoch results in `<out>/log.jsonl`.
