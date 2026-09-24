# Rational basis functions for continuous-kernel graph convolutions

A minimal, self-contained repository for the experiments in
`paper/rational_basis_draft.tex`: replacing the fixed B-spline basis of
**SplineCNN** (Fey et al., CVPR 2018) by **learnable rational (safe-Padé)
basis functions**, evaluated on

* **PascalVOC-Keypoints** graph matching with Deep Graph Matching Consensus
  (DGMC, Fey et al., ICLR 2020), and
* **FAUST** shape correspondence (the SplineCNN reference task), and
* **N-Caltech101** event-camera object recognition with the AEGNN network
  (Schaefer et al., CVPR 2022), whose SplineConv layers are the operator's
  main application in event-based vision.

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
  large.py               basis-first aggregation for graphs with ~1e7 edges (large=True in both convs)
  triton_basis.py        fused Triton kernels for the multivariate rational basis (default on CUDA)
  spline_cnn.py          SplineCNN backbone stack (DGMC ψ networks)
  dgmc.py, data.py       DGMC model and pair datasets (from rusty1s/deep-graph-matching-consensus)
  face_to_edge.py        FaceToEdge tolerant of <3-keypoint graphs (PyG ≥ 2.4 asserts)
  ply.py                 plyfile-based .ply reader for FAUST (PyG's needs openmesh)
experiments/
  pascal_voc.py          Experiment 1: PascalVOC-Keypoints / DGMC
  faust.py               Experiment 2: FAUST shape correspondence
  ncaltech101.py         Experiment 3: N-Caltech101 / AEGNN recognition network (also N-Cars)
  aegnn_net.py           AEGNN GraphRes with pluggable conv (PyG SplineConv, ours, rational, PointNet)
  event_utils.py         AEGNN event pre-processing (median 50 ms window, fixed 25k events, beta time)
  prepare_ncaltech101.py one-time download (5.9 GB, Gehrig et al. split) + pre-processing
  prepare_ncars.py       same for N-Cars (Prophesee .dat; the data is behind a request form)
  bench_kernels.py       layer / basis speed and memory vs. PyG's fused torch_spline_conv kernels
  plot_basis.py          plots learned 2-D basis functions vs. their initialization (paper/figures/)
  backbones.py           --backbone/--rational_basis/--init/... flags shared by both scripts
  wandb_util.py          optional wandb logging (grad/update/param norms, losses, accuracies)
  prepare_pascal_voc.py  one-time dataset download (annotations via the Internet Archive) + VGG16 features
  prepare_faust.py       one-time placement + processing of MPI-FAUST.zip
slurm/                   configs.sh (every named configuration), sbatch runners, submit.sh
results/
  analyze.py             aggregates results/logs into the tables below (mean ± std, Welch t-tests)
  bench_kernels.txt      output of experiments/bench_kernels.py (RTX 4070 Ti Super, 12.4M edges)
  logs/pascal_voc/       95 training logs (19 configs × 5 seeds)
  logs/faust/            42 training logs (14 configs × 3 seeds)
  logs/ncaltech101/      54 training logs (20 configs, mostly 3 seeds)
  logs/ncars/            18 training logs (6 configs x 3 seeds)
tests/                   30 pytest tests (basis fits, PCA init, vp gain, conv / kernel equivalences, DGMC)
paper/                   rational_basis_draft.tex / .pdf, figures/ (learned PascalVOC bases)
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

**N-Caltech101** (Orchard et al., CC BY 4.0; the training / validation /
test split of Gehrig et al., ICCV 2019, 5.9 GB):

```bash
python experiments/prepare_ncaltech101.py --download   # -> data/NCaltech101, ~2.7 GB processed
```

Per-batch radius graphs need `torch_cluster` (`pip install torch_cluster`;
for recent torch/CUDA combinations without wheels, build from source with
`FORCE_CUDA=1 pip install --no-build-isolation git+https://github.com/rusty1s/pytorch_cluster`).
`--backbone pyg_spline` (the original fused kernel) additionally needs
`torch_spline_conv`; without it use `--backbone spline`, our implementation of
the same operator (identical to 1e-7, faster and leaner on these graphs).

**N-Cars** must be requested from Prophesee
(https://www.prophesee.ai/2018/03/13/dataset-n-cars/); then
`python experiments/prepare_ncars.py --src /path/to/extracted` and
`--dataset ncars` on `ncaltech101.py` (untested: we did not have the data).

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
```

# Experiment 3 — N-Caltech101 / AEGNN (30 epochs, ~7 min/epoch (spline) on an RTX 4070 Ti; seeds 0–2)
python experiments/ncaltech101.py --backbone spline                                        # AEGNN recognition net, k=2 (K=8)
python experiments/ncaltech101.py --backbone rational --rational_basis multivariate \
       --degrees 8 6 --init pca --vp --num_bases 8                                         # mv, K=8
python experiments/ncaltech101.py --backbone pointnet                                      # Jeziorek et al. replacement
```

Defaults are the exact paper settings: PascalVOC — DGMC with ψ₁: 1024→256,
ψ₂: 128→128, 2 conv layers each, 10 consensus steps, Adam lr 1e-3, batch
512, 15 epochs, 1000 test samples per category; FAUST — 6 conv layers
(32, 64×5), ELU, Lin 256, dropout 0.5, Adam lr 0.01 → 0.001 at epoch 61,
batch 1, 100 epochs, gradient-norm clipping 1.0, `add` aggregation (the PyG
reference; `--aggr mean` is the paper's operator); N-Caltech101 — the AEGNN
recognition network (7 SplineConv layers, kernel size 2, channels
1 8 16 16 16 32 32 32, BatchNorm, ELU, two voxel max-poolings, no root
weight / bias, mean aggregation), 25 000 events per sample from the 50 ms
window before the median event, radius graph r = 5 with at most 32
neighbours, time scaled by β = 0.5e-5/µs, Adam lr 1e-3, batch 16,
cross-entropy, lr / 10 after epoch 20, 30 epochs. Add `--wandb` for Weights &
Biases logging, `--seed N` for the seed.

### Slurm

```bash
sbatch slurm/prep_pascal_voc.sbatch                # once
slurm/submit.sh voc                                 # all paper configs × seeds 0–4
slurm/submit.sh faust                               # all paper configs × seeds 0–2
slurm/submit.sh ncal                                # all N-Caltech101 configs × seeds 0–2
slurm/submit.sh voc mv_K4 mv_K6                     # a subset
SEEDS="5 6" WANDB=1 slurm/submit.sh faust faust_mv_K8
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

**N-Caltech101 / AEGNN**, test accuracy after 30 epochs of the AEGNN
recognition network (7 convolutions, kernel size 2, channels
1 8 16 16 16 32 32 32, mean aggregation unless noted), mean ± std over 3
seeds. `conv` counts the parameters of the seven convolutions; PointNet is
the SplineConv replacement of Jeziorek et al. (2023), `max_j W [x_j, u_ij]`.

| config (`configs.sh` name) | aggr | conv params | final acc |
|---|---|---|---|
| SplineConv k=2, K=8 (`ncal_spline`, AEGNN as published) | mean | 25.7k | 42.68 ± 0.81 |
| SplineConv k=2, PyG fused kernel (`ncal_pyg_spline`, 1 seed) | mean | 25.7k | 41.41 |
| product rational k=2, hat init (`ncal_rational_k2`, 1 seed) | mean | 26.1k | 43.54 |
| **multivariate K=4, PCA+vp (`ncal_mv_K4`)** | mean | 19.8k | **48.78 ± 0.77** |
| **multivariate K=8, PCA+vp (`ncal_mv_K8`)** | mean | 39.6k | **48.57 ± 1.98** |
| PointNet conv (`ncal_pointnet`) | max | 3.7k | 51.46 ± 1.24 |
| PointNet conv (`ncal_pointnet_mean`) | mean | 3.7k | 41.01 ± 1.15 |
| SplineConv k=2 (`ncal_spline_max`) | max | 25.7k | 52.71 ± 1.70 |
| **multivariate K=8 (`ncal_mv_K8_max`)** | max | 39.6k | **54.51 ± 1.40** |
| SplineConv k=2 + flip/shift augmentation (`ncal_spline_aug`) | mean | 25.7k | 47.14 ± 0.63 |
| **multivariate K=8 + augmentation (`ncal_mv_K8_aug`)** | mean | 39.6k | **54.32 ± 0.18** |
| PointNet conv + augmentation (`ncal_pointnet_aug`) | max | 3.7k | 54.84 ± 0.24 |
| SplineConv k=2 + augmentation (`ncal_spline_max_aug`) | max | 25.7k | 54.72 ± 0.63 |
| **multivariate K=8 + augmentation (`ncal_mv_K8_max_aug`)** | max | 39.6k | **57.27 ± 0.46** |
| *trained to convergence* (`--schedule plateau`: lr/10 on validation plateaus, stop after the 2nd; 51-92 epochs) | | | |
| SplineConv k=2 + aug (`ncal_spline_aug_conv`) | mean | 25.7k | 47.35 ± 0.15 |
| multivariate K=8 + aug (`ncal_mv_K8_aug_conv`) | mean | 39.6k | 55.05 ± 0.58 |
| PointNet conv + aug (`ncal_pointnet_aug_conv`) | max | 3.7k | 54.99 ± 0.84 |
| SplineConv k=2 + aug (`ncal_spline_max_aug_conv`) | max | 25.7k | 56.00 ± 0.92 |
| **multivariate K=8 + aug (`ncal_mv_K8_max_aug_conv`)** | max | 39.6k | **57.55 ± 0.42** |

Welch t-tests: `mv_K4` vs `spline` +6.11 (p = 0.001), `mv_K8` vs `spline`
+5.90 (p = 0.023), `mv_K8_aug` vs `spline_aug` +7.18 (p = 0.001),
`mv_K8_max` vs `spline_max` +1.80 (p = 0.23), `mv_K8_max` vs `pointnet`
+3.05 (p = 0.048), `pointnet_mean` vs `pointnet` −10.45 (p < 0.001),
`spline_max` vs `spline` +10.03 (p = 0.003), `mv_K8_max_aug` vs
`spline_max_aug` +2.55 (p = 0.006) and vs `pointnet_aug` +2.43 (p = 0.004);
to convergence: `mv_K8_aug_conv` vs `spline_aug_conv` +7.70 (p = 0.001),
`mv_K8_max_aug_conv` vs `spline_max_aug_conv` +1.55 (p = 0.08) and vs
`pointnet_aug_conv` +2.57 (p = 0.019). Training to convergence changes no
configuration by more than 1.3 points (all p > 0.1 vs the 30-epoch runs).

Reading: with AEGNN's protocol the rational basis adds about 6 points over
SplineConv at equal K (and K=4 does so with 23% fewer conv parameters); the
gain persists under augmentation (+7). PointNet's advantage over SplineConv
(Jeziorek et al.) is entirely its max aggregation: with mean aggregation it
drops to SplineConv level, and SplineConv / the rational basis with max
aggregation gain 10 / 6 points. The absolute numbers are below the 66.8%
reported for AEGNN: the released code has no training script or
augmentation, so the published regularisation could not be reproduced; the
fidelity check with PyG's own SplineConv kernel lands at the same level as
our implementation. The rational runs use the fused Triton kernels
(`rational_cnn/triton_basis.py`, validated against the eager evaluation on
`ncal_mv_K8_triton`: 47.85 vs 48.48 with the same seed), which make a
rational epoch ~1.2x a SplineConv epoch.

**N-Cars** (Prophesee; car vs. background, 100 ms samples), test accuracy
after 30 epochs of the same network with AEGNN's N-Cars settings (10 000
events, r = 3, at most 32 neighbours, batch 64, 120 x 100 px), mean ± std
over 3 seeds; 10% of the training sequences are held out for validation.

| config (`configs.sh` name) | aggr | conv params | final acc |
|---|---|---|---|
| SplineConv k=2, K=8 (`ncars_spline`) | mean | 25.7k | 87.83 ± 0.25 |
| PointNet conv (`ncars_pointnet`) | max | 3.7k | 87.17 ± 0.24 |
| **multivariate K=4 (`ncars_mv_K4`)** | mean | 19.8k | **90.24 ± 0.39** |
| **multivariate K=8 (`ncars_mv_K8`)** | mean | 39.6k | **90.49 ± 0.05** |
| SplineConv k=2 (`ncars_spline_max`) | max | 25.7k | 89.81 ± 0.27 |
| **multivariate K=8 (`ncars_mv_K8_max`)** | max | 39.6k | **91.00 ± 0.43** |
| *AEGNN paper* | | | *94.5* |

Welch t-tests: `mv_K8` vs `spline` +2.66 (p = 0.002), `mv_K4` vs `spline`
+2.41 (p = 0.002), `mv_K8_max` vs `spline_max` +1.19 (p = 0.021),
`spline_max` vs `spline` +1.97 (p = 0.001), `pointnet` vs `spline` −0.66
(p = 0.03). On N-Cars the rational basis with mean aggregation already beats
SplineConv with max aggregation (+0.68, p = 0.04), and PointNet is the weakest
operator.

## Learned basis functions

`python experiments/pascal_voc.py <config flags> --save_model results/models/x.pt`
stores the trained DGMC model; `python experiments/plot_basis.py
results/models/x.pt --layers psi_1.convs.0,psi_1.convs.1 --out fig.pdf` plots
every basis function on the pseudo-coordinate square next to its
initialization (`paper/figures/basis_voc_{rational_mv_k3,mv_K6,mv_K4}.pdf`).
`--nmt` reads the `psi_final.pt` written by the NMT training script.

## Kernel performance

`experiments/bench_kernels.py` times one AEGNN-scale layer (16 x 25 000
nodes, 12.4M edges, D = 3, kernel size 2 so K = 8 hats, mean aggregation, no
root weight / bias) on an RTX 4070 Ti Super; full output in
`results/bench_kernels.txt`. The rational layers evaluate the basis with the
Triton kernels of `rational_cnn/triton_basis.py` and aggregate basis-first
(`rational_cnn/large.py`); PyG's SplineConv uses the fused CUDA kernels of
`torch_spline_conv` (`--backbone pyg_spline`).

| layer, 32 → 32 channels | fwd | fwd + bwd | peak mem |
|---|---|---|---|
| PyG SplineConv, `torch_spline_conv` fused kernels | 97 ms | 624 ms | 5.5 GB |
| our B-spline, dense basis + basis-first aggregation | 172 ms | 348 ms | 5.2 GB |
| rational, Triton basis, K=8, degrees (8, 6) | 152 ms | 444 ms | 5.7 GB |
| rational, Triton basis, K=8, degrees (5, 4) | 151 ms | 435 ms | 5.7 GB |
| rational, Triton basis, K=4, degrees (8, 6) | 76 ms | 222 ms | 5.1 GB |
| rational, `torch.compile`d chunked basis, K=8 | 212 ms | 583 ms | 5.7 GB |
| rational, eager chunked basis, K=8 | 395 ms | 951 ms | 5.7 GB |

| basis only, 12.4M edges | time | peak mem |
|---|---|---|
| `torch_spline_conv.spline_basis`, forward (K=8) | 8.9 ms | 1.1 GB |
| Triton rational basis, forward (K=8, degrees (8, 6)) | 4.4 ms | 0.7 GB |
| Triton rational basis, forward + backward | 19.2 ms | 1.2 GB |

With the fused kernels the basis evaluation is under 5% of the layer time
and its degree does not matter; the cost of a rational layer is the
neighbourhood aggregation, exactly as for B-splines. PyG's forward pass is
1.6x faster (its `spline_weighting` never forms the per-basis aggregates
`Y[i, p]`), but its per-edge backward is slow, so a training step of the
rational K=8 layer is 1.4x faster than PyG's SplineConv (K=4: 2.8x) and on
par with it at 16 channels. In the N-Caltech101 runs on the same GPU an
epoch takes 264 s (our B-spline), 316 s (rational K=8, Triton), 164 s
(rational K=4) and 1006 s (rational K=8, `torch.compile` basis).

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
  when `K ≠ k^D`; `--vp` rescales the kernel weights by
  `sqrt(E_u Σ_p hat_p(u)² / E_u Σ_p B_p(u)²)` so that the message variance no
  longer depends on the shape of the basis (KAT-style). Note that it equals
  SplineConv's variance only for `K = k^D`: the uniform weight bound
  `1/sqrt(K·C_in)` leaves a factor `k^D / K` (2 for the K=4 event-camera
  configs, ≈16 for FAUST K=8, measured 2.09 / 18.1; harmless after the
  BatchNorm of the AEGNN network but relevant on FAUST). The denominator is never started at exactly
  zero (`∂|Q|/∂b = 0` there would freeze it). `random`, `constant`, `gauss`,
  `cheb` are the ablations of the paper.
* **Controls**: `--rational_basis mlp` (a filter-generating MLP with the same
  `Θ_p`), `--degrees 5 4` with the multivariate basis (degree control),
  `--pyg_init`, `--clip`, `--aggr`.

See `paper/rational_basis_draft.pdf` for the full description, the
initialization analysis and the discussion of the SplineCNN reproduction gap
on FAUST (`add` vs `mean` aggregation, gradient clipping).

## License

MIT. `rational_cnn/dgmc.py`, `data.py` and `spline_cnn.py` are derived from
[rusty1s/deep-graph-matching-consensus](https://github.com/rusty1s/deep-graph-matching-consensus)
(MIT, Matthias Fey). The FAUST dataset itself is *not* included and may not
be redistributed.
