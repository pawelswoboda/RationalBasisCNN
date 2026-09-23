# Rational basis functions for continuous-kernel graph convolutions

A minimal, self-contained repository for the experiments in
`paper/rational_basis_draft.tex`: replacing the fixed B-spline basis of
**SplineCNN** (Fey et al., CVPR 2018) by **learnable rational (safe-Padé)
basis functions**, evaluated on

* **PascalVOC-Keypoints** graph matching with Deep Graph Matching Consensus
  (DGMC, Fey et al., ICLR 2020), and
* **FAUST** shape correspondence (the SplineCNN reference task).

Every layer computes the SplineConv operator
`x'_i = Θ_root x_i + □_{j∈N(i)} (Σ_p B_p(u_ij) Θ_p) x_j + b`; only the basis
`{B_p}` differs. Because rational functions need not be localised, the number
of basis functions `K` is decoupled from the spline grid `k^D`: `K = 4–8`
non-separable rational functions match or beat `K = 25–125` B-spline hats at
2–5× fewer parameters.

## Layout

```
rational_cnn/            the package
  rational.py            RationalBasis1D / RationalBasis (tensor product), MultivariateRationalBasis
                         (total-degree polynomials in all D coordinates, free K), MLPBasis (control),
                         RationalConv, RationalCNN; hat-fit / PCA-of-hats / vp initialization
  bspline.py             BSplineConv: pure-PyTorch SplineConv (no torch-spline-conv needed)
  spline_cnn.py          SplineCNN backbone stack (DGMC ψ networks)
  dgmc.py, data.py       DGMC model and pair datasets (from rusty1s/deep-graph-matching-consensus)
  face_to_edge.py        FaceToEdge tolerant of <3-keypoint graphs (PyG ≥ 2.4 asserts)
  ply.py                 plyfile-based .ply reader for FAUST (PyG's needs openmesh)
  hpatches.py            HPatches loading, Shi-Tomasi detector, distractors, transfer arms
  spair.py               SPair71k (VGG16 keypoint graphs + pair tables), SPair71kPairs, MatchingAccuracy
experiments/
  pascal_voc.py          Experiment 1: PascalVOC-Keypoints / DGMC
  faust.py               Experiment 2: FAUST shape correspondence
  spair71k.py            Experiment 3: SPair-71k keypoint matching / DGMC
  backbones.py           --backbone/--rational_basis/--init/... flags shared by both scripts
  wandb_util.py          optional wandb logging (grad/update/param norms, losses, accuracies)
  prepare_pascal_voc.py  one-time dataset download (annotations via the Internet Archive) + VGG16 features
  prepare_faust.py       one-time placement + processing of MPI-FAUST.zip
  prepare_spair71k.py    one-time VGG16 feature extraction + pair tables for SPair-71k
  hpatches_transfer.py   out-of-task transfer of a trained refinement, measured on HPatches
slurm/                   configs.sh (every named configuration), sbatch runners, submit.sh
results/
  analyze.py             aggregates results/logs into the tables below (mean ± std, Welch t-tests)
  logs/pascal_voc/       95 training logs (19 configs × 5 seeds)
  logs/faust/            42 training logs (14 configs × 3 seeds)
  logs/spair71k/         written by slurm/spair71k.sbatch (no runs yet)
tests/                   42 pytest tests (basis fits, PCA init, vp gain, conv equivalences, DGMC, SPair-71k, HPatches)
paper/                   rational_basis_draft.tex / .pdf
```

## Installation

```bash
pip install torch torch_geometric torchvision scipy numpy plyfile pytest   # see requirements.txt
pip install -e .            # or run scripts from the repo root; they add it to sys.path themselves
python -m pytest            # ~2 min on CPU
```

Tested with torch 2.12 / PyG 2.8 / CUDA 13. No PyG C++ extensions
(`torch-spline-conv`, `pyg-lib`, `torch-scatter`) are needed.

## Data

**PascalVOC-Keypoints** (downloaded automatically, ~2 GB + VGG16 feature
extraction on GPU, ~20 min):

```bash
python experiments/prepare_pascal_voc.py           # -> data/PascalVOC
```

PyG's original annotation URL is dead; the script fetches
`voc2011_keypoints_Feb2012.tgz` from the Internet Archive instead.

**FAUST** must be requested from http://faust.is.tue.mpg.de/ (the license
forbids redistribution). Then:

```bash
python experiments/prepare_faust.py --src /path/to/MPI-FAUST.zip    # or the unzipped MPI-FAUST/ dir
```

**SPair-71k** (Min et al., 2019; 18 categories, 1,800 images, 70,958 annotated
pairs). Extract `SPair-71k.tar.gz` to `data/SPair-71k` (on PC2:
`bash hpc/download_spair71k.sh`), then build the graphs once:

```bash
python experiments/prepare_spair71k.py              # -> data/SPair71k/processed
```

Node features follow PyG's PascalVOCKeypoints exactly (box grown to all
keypoints + 16 px, 256×256 crop, VGG16 relu4_2 + relu5_1). The benchmark
protocol follows pygmtools' SPair71k: the fixed trn/val/test pairs of the
`large` layout, no difficulty filtering, and only keypoints visible in both
images.

Use `--root <dir>` on any script to keep the data elsewhere than `data/`.

## Running the experiments

All named configurations are in `slurm/configs.sh`; `config_args <name>`
prints the flags. The headline runs:

```bash
# Experiment 1 — PascalVOC (15 epochs, ~7–10 s/epoch on an RTX 4090; seeds 0–4 in the paper)
python experiments/pascal_voc.py --backbone spline                                         # SplineCNN k=5 (K=25)
python experiments/pascal_voc.py --backbone rational --rational_basis multivariate \
       --degrees 8 6 --kernel_size 3                                                       # mv, hat init, K=9
python experiments/pascal_voc.py --backbone rational --rational_basis multivariate \
       --degrees 8 6 --init pca --vp --num_bases 4                                         # mv, PCA+vp, K=4

# Experiment 2 — FAUST (100 epochs, ~7–10 s/epoch; seeds 0–2 in the paper)
python experiments/faust.py --backbone spline --aggr mean                                  # best SplineCNN reproduction
python experiments/faust.py --backbone rational --rational_basis multivariate \
       --degrees 8 6 --init pca --vp --num_bases 8                                         # mv, K=8

# Experiment 3 — SPair-71k (same DGMC and backbone flags as PascalVOC)
python experiments/spair71k.py --backbone spline                                           # SplineCNN k=5 (K=25)
python experiments/spair71k.py --backbone rational --rational_basis multivariate \
       --degrees 8 6 --init pca --vp --num_bases 4 --num_workers 6                         # mv, PCA+vp, K=4
```

SPair-71k uses the PascalVOC defaults (not tuned for SPair) over all 53,340
training pairs per epoch. The target keypoints are shuffled (randomly for
training, with a fixed permutation per pair for evaluation) because the
annotated order is the same in both images. After every epoch it reports
per-category accuracy on val and all test pairs, both pair-averaged (the
benchmark convention, used as the headline) and keypoint-weighted (the
PascalVOC convention; 1,860 pairs have a single, trivially matched keypoint).
The final line gives the last-epoch test accuracy, the best, and the test
accuracy at the best validation epoch.

Defaults are the exact paper settings: PascalVOC — DGMC with ψ₁: 1024→256,
ψ₂: 128→128, 2 conv layers each, 10 consensus steps, Adam lr 1e-3, batch
512, 15 epochs, 1000 test samples per category; FAUST — 6 conv layers
(32, 64×5), ELU, Lin 256, dropout 0.5, Adam lr 0.01 → 0.001 at epoch 61,
batch 1, 100 epochs, gradient-norm clipping 1.0, `add` aggregation (the PyG
reference; `--aggr mean` is the paper's operator). Add `--wandb` for Weights &
Biases logging, `--seed N` for the seed.

### Slurm

```bash
sbatch slurm/prep_pascal_voc.sbatch                # once
slurm/submit.sh voc                                 # all paper configs × seeds 0–4
slurm/submit.sh faust                               # all paper configs × seeds 0–2
slurm/submit.sh voc mv_K4 mv_K6                     # a subset
SEEDS="5 6" WANDB=1 slurm/submit.sh faust faust_mv_K8
sbatch slurm/prep_spair71k.sbatch                  # once, SPair-71k graphs
slurm/submit.sh spair                               # PascalVOC paper configs on SPair-71k × seeds 0–4
DRY_RUN=1 slurm/submit.sh spair mv_K4               # print the jobs, submit nothing
python results/analyze.py                           # tables + Welch t-tests from results/logs
```

Set `PYTHON=/path/to/python` if `python` is not the right interpreter on the
compute nodes; edit the `#SBATCH --partition` lines for your cluster.

## Results (from `results/analyze.py`)

**PascalVOC-Keypoints**, mean keypoint-matching accuracy over 20 categories
at the final epoch, mean ± std over 5 seeds.

| config (`configs.sh` name) | K | params | final acc | best acc |
|---|---|---|---|---|
| SplineCNN k=5 (`spline`) | 25 | 9.50M | 72.00 ± 0.39 | 73.00 ± 0.40 |
| SplineCNN k=3 (`spline_k3`) | 9 | 3.74M | 72.06 ± 0.85 | 72.78 ± 0.40 |
| SplineCNN k=2 (`spline_k2`) | 4 | 1.93M | 70.94 ± 0.63 | 71.48 ± 0.40 |
| product rational k=5 (`rational`) | 25 | 9.50M | 72.24 ± 0.67 | 72.88 ± 0.48 |
| product rational k=3 (`rational_k3`) | 9 | 3.74M | 72.62 ± 0.41 | 73.26 ± 0.18 |
| **multivariate, hat init, k=3 (`rational_mv_k3`)** | 9 | 3.74M | **73.76 ± 0.81** | **74.12 ± 0.48** |
| multivariate, PCA+vp, K=4 (`mv_K4`) | 4 | 1.94M | 72.92 ± 0.36 | 73.78 ± 0.41 |
| multivariate, PCA+vp, K=6 (`mv_K6`) | 6 | 2.66M | 73.10 ± 0.41 | 74.06 ± 0.37 |
| multivariate, PCA+vp, K=9 (`mv_K9`) | 9 | 3.74M | 73.16 ± 0.47 | 73.72 ± 0.19 |
| multivariate, PCA+vp, K=16 (`mv_K16`) | 16 | 6.26M | 72.26 ± 0.42 | 73.26 ± 0.42 |
| MLP basis K=4 / K=9 (`mlp_K4`, `mlp_K9`) | 4 / 9 | 1.94M / 3.74M | 72.12 ± 0.32 / 71.46 ± 0.64 | |
| multivariate (5,4), k=5 (`rational_mv_d54`, degree control) | 25 | 9.51M | 72.04 ± 0.94 | |

Welch t-tests: `rational_mv_k3` vs `spline` +1.76 (p = 0.005); `mv_K4` vs
`spline` +0.92 (p = 0.005), vs `mlp_K4` +0.80 (p = 0.006), vs `spline_k2`
(equal K and parameters) +1.98 (p = 0.001).

**FAUST**, exact vertex-correspondence accuracy after 100 epochs, mean ± std
over 3 seeds. All rows use gradient clipping 1.0 and `add` aggregation
unless noted.

| config | K | params | final acc |
|---|---|---|---|
| **multivariate K=8 (`faust_mv_K8`)** | 8 | 1.97M | **99.41 ± 0.04** |
| **multivariate K=4 (`faust_mv_K4`)** | 4 | 1.89M | **99.38 ± 0.09** |
| product rational k=3 (`faust_rational_k3`) | 27 | 2.30M | 99.24 ± 0.08 |
| *SplineCNN paper (CUDA kernel)* | 125 | 4.1M | *99.20* |
| multivariate K=4, mean aggr (`faust_mv_K4_mean`) | 4 | 1.89M | 98.93 ± 0.10 |
| SplineCNN k=5, mean aggr (`faust_spline_mean`, best reproduction) | 125 | 4.11M | 98.71 ± 0.08 |
| MLP basis K=8 (`faust_mlp_K8`) | 8 | 1.96M | 98.21 ± 1.11 |
| SplineCNN k=5, add aggr (`faust_spline`) | 125 | 4.11M | 97.23 ± 0.51 |
| SplineCNN k=3 (`faust_spline_k3`) | 27 | 2.30M | 83.6 ± 15.6 |
| multivariate K=16 / K=27 (`faust_mv_K16`, `faust_mv_K27`) | 16 / 27 | 2.13M / 2.34M | 70.4 ± 25.2 / 19.0 ± 32.8 |
| SplineCNN k=5, add, no clip (`faust_spline_noclip`, literal PyG example) | 125 | 4.11M | 29.1 ± 50.3 |
| SplineCNN k=2 (`faust_spline_k2`) | 8 | 1.95M | 3.4 ± 3.3 |

## Method summary

* **Product basis** (`--rational_basis product`): `B_p(u) = Π_d r_{p_d}(u_d)`
  with per-dimension safe-Padé functions `r(u) = P(t) / (1 + |Q(t)|)`,
  `t = 2u − 1`, Chebyshev features, degrees (5, 4). Same `K = k^D` as the hats.
* **Multivariate basis** (`--rational_basis multivariate`): each `B_p` is a
  single non-separable rational function of all coordinates with numerator /
  denominator of total degree (8, 6); `K` is free (`--num_bases`).
* **Initialization** (`--init`): `spline` fits each rational function to its
  B-spline hat (LS → Adam → L-BFGS, float64) so the layer starts as a
  SplineConv; `pca` fits the K leading principal components of the hat basis
  when `K ≠ k^D`; `--vp` rescales the kernel weights so the message variance
  matches SplineConv (KAT-style). The denominator is never started at exactly
  zero (`∂|Q|/∂b = 0` there would freeze it). `random`, `constant`, `gauss`,
  `cheb` are the ablations of the paper.
* **Controls**: `--rational_basis mlp` (a filter-generating MLP with the same
  `Θ_p`), `--degrees 5 4` with the multivariate basis (degree control),
  `--pyg_init`, `--clip`, `--aggr`.

See `paper/rational_basis_draft.pdf` for the full description, the
initialization analysis and the discussion of the SplineCNN reproduction gap
on FAUST (`add` vs `mean` aggregation, gradient clipping).

## HPC (PC2 Noctua 2)

This clone is wired into PC2 Noctua 2: `hpc/env.sh` moves the dataset root,
caches, wandb and the conda environment off the quota-limited `$HOME` onto
project storage, and the `slurm/` jobs request the `gpu_h100` partition under
account `hpc-prf-llmrout`. See `docs/hpc_runbook.md`.

```bash
source hpc/env.sh
bash hpc/create_environment.sh     # conda env on group storage + pytest
sbatch slurm/prep_pascal_voc.sbatch
```

The `--root` defaults fall back to `<repo>/data` when `$RBCNN_DATA_ROOT` is
unset, so the repository behaves exactly as documented above elsewhere.

## License

MIT. `rational_cnn/dgmc.py`, `data.py` and `spline_cnn.py` are derived from
[rusty1s/deep-graph-matching-consensus](https://github.com/rusty1s/deep-graph-matching-consensus)
(MIT, Matthias Fey). The FAUST dataset itself is *not* included and may not
be redistributed.
