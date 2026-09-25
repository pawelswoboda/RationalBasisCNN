# NMT with a VGG16 backbone and rational basis functions

This directory holds the second host pipeline of the keypoint-matching
experiments: the **Normalized Matching Transformer** (NMT, Pourhadi & Swoboda,
ICPR 2026, [arXiv:2503.17715](https://arxiv.org/abs/2503.17715)). Its
geometry-aware keypoint refinement is a SplineCNN, and the experiment replaces
only the B-spline basis of those SplineConv layers with the rational bases of
`rational_cnn` (the package at the repository root). The matching
transformer, the losses and the optimiser settings stay as NMT defines them.
Two things differ from the NMT paper and apply to every configuration alike:
the SwinV2 backbone is replaced by the VGG16 of DGMC at 256 x 256 pixels, and
runs stop after 7 of the 12 scheduled epochs (see [Running](#running)).

The code is NMT at upstream commit `9370239` ("major fixes + refactor",
2026-04-21) plus the changes listed below. The exact diff of the seven
modified source files is in [`upstream_changes.patch`](upstream_changes.patch),
which applies to that commit with `git apply`. The code is reduced to what the
VGG16 runs import (see [What is included](#what-is-included)).

## What changed relative to upstream NMT

Two switches, both defaulting to the upstream behaviour:

1. **`cfg.BACKBONE`** selects the visual backbone: `"swin"` (the SwinV2 of the
   NMT paper) or `"vgg16"` / `"vgg16_bn"` (14.7 M parameters, the relu4_2 +
   relu5_1 extractor of SplineCNN and DGMC). All runs here use
   `vgg16`, so that NMT sits at the same backbone scale as DGMC.
2. **`cfg.SPLINE_CNN.conv`** selects the continuous-kernel convolution:
   `"spline"` (B-spline hats, `rational_cnn.BSplineConv`, the same operator as
   `torch_geometric.nn.SplineConv` without its C++ extension) or `"rational"`
   (`rational_cnn.RationalConv`). The two differ only in the basis.

| file | change |
|---|---|
| `model/sconv_archs.py` | `make_conv()` builds `BSplineConv` or `RationalConv` from `cfg.SPLINE_CNN` instead of PyG's `SplineConv(dim=2, kernel_size=5, aggr="max")` |
| `utils/config.py` | new keys `SPLINE_CNN.*`, `BACKBONE`, `IMAGE_SIZE`, `TRAIN.max_epochs`, `keep_last_checkpoint_only`, `compile` (defaults reproduce upstream) |
| `utils/backbone.py` | `VisualBackbone` (SwinV2 or VGG16 behind one `extract()` interface), a plain `VGG16` class, `backbone_params` on `VGG16_base` |
| `model/nmt.py` | uses `VisualBackbone`; `feature_align` takes `cfg.IMAGE_SIZE` instead of a hard-coded 384 |
| `utils/build_graphs.py` | edge pseudo-coordinates divide by `cfg.IMAGE_SIZE` instead of 384 |
| `train_eval.py` | dataset resize from `cfg.IMAGE_SIZE`; stop after `cfg.TRAIN.max_epochs`; optional pruning of old checkpoints; optional `torch.compile` of the two nGPT decoders (`cfg.compile`) |
| `data/SPair71k.py` | accepts the official `:` separator in pair-annotation names as well as `_` |

New files: `experiments/{spair,voc}_vgg16.json` (the two base configs),
`slurm/` (config generator and launch scripts), `hpc/` and
`environment-hpc.yml` (environment), `tests/`, `conftest.py`, `pytest.ini`,
`results/analyze.py` and `upstream_changes.patch`.

### Why the changes are easy to get wrong

* **Three image sizes must agree.** `cfg.IMAGE_SIZE` drives the dataset resize
  (`train_eval.py`), the coordinate frame of `utils/feature_align.py`
  (`model/nmt.py`) and the divisor of the edge pseudo-coordinates
  (`utils/build_graphs.py`). The pseudo-coordinates `0.5 + 0.5 * dx / size`
  must lie in [0, 1] for the kernel. A mismatch silently shrinks or overflows
  the domain instead of raising.
* **`VGG16_base` needs `backbone_params`**, because `train_eval.py` trains the
  backbone in its own optimizer group at 0.03x the learning rate.
* **`AdaptiveMaxPool2d(1, 1)`** sets `return_indices=True`, so the global
  feature is taken from the `(values, indices)` tuple.
* **SPair-71k file names.** The upstream loader rewrote `:` to `_`, but the
  official release uses `:` exactly as its Layout files do. The loader now uses
  whichever exists.
* **A free number of rational bases needs `init: "pca"`.** Hat-fitting
  initialisation (`init: "spline"`) needs one rational function per B-spline
  hat, i.e. `num_bases == kernel_size**dim`. The wrong combination raises at
  construction, and `tests/test_backbone_and_conv.py` pins that.

## What is included

Every file that `train_eval.py`, the tests, `slurm/` and `results/` import.
The Python modules (`train_eval.py`, `eval.py`, `model/`, `data/`, `utils/`),
the tests and the configs are byte-identical to the code that produced the
logged runs. `slurm/`, `hpc/` and `results/analyze.py` were changed when they
moved into this repository, but only in comments, messages and relative paths
(the environment script now installs `rational_cnn` from `..`), and
`results/analyze.py` now reads `../results/logs/nmt_<dataset>` by default.
`slurm/make_config.py` regenerates the settings of all 150 logged runs
exactly. The SwinV2, ViT, GMT and WILLOW modules are part of the set because
`utils/__init__.py`, `utils/backbone.py` and `data/data_loader_multigraph.py`
import them at module level. The SwinV2 weights are not included, and the
`swin` path was not run for this project.

Left out from upstream: its README, `.gitignore` (the repository's covers
`nmt/`) and `misc/` figures, `environment.yml`, `download_data.sh`,
`run_script.txt`, the package `__init__.py` at its root,
`experiments/spair.json` (the SwinV2 SPair-71k config), the modules nothing
imports (`utils/decorators.py`, `dup_stdout_manager.py`, `latex_utils.py`,
`mma_metrics.py`, `vit_hyperspherical.py`) and the run artefacts in `errors/`.
`experiments/voc_basic.json` stays, since both VGG16 configs inherit from it.

## Setup

Everything runs from this directory, because NMT imports `utils`, `model` and
`data` as top-level packages and reads its configs by relative path.

```bash
cd nmt
conda env create -f environment-hpc.yml   # torch 2.12 + CUDA 13, PyG 2.8
conda activate nmt
pip install --no-deps -e ..               # rational_cnn from the repository root
python -m pytest                          # 11 tests, CPU, no dataset needed
```

On PC2 Noctua 2, `hpc/env.sh` moves the conda environment, run outputs and
every cache off the quota-limited `$HOME`, and `hpc/create_environment.sh`
builds the environment from `environment-hpc.yml`, installs `rational_cnn`
editable from `..`, pre-fetches the VGG16 weights so that concurrent jobs do
not race on the download, and runs the tests:

```bash
source hpc/env.sh
bash hpc/create_environment.sh       # login node
```

The environment has no `pytorch_spline_conv`, `torch-scatter` or `timm`. The
Swin layers `timm` would provide are vendored in `utils/timmLayers/`. The
VGG16 ImageNet weights come from torchvision and are downloaded into
`$TORCH_HOME` on first use, so pre-fetch them where compute nodes have no
internet access.

**Speed.** `"compile": true` in a config wraps the two normalized-transformer
decoders in `torch.compile(dynamic=True)`. They consist of thousands of tiny
kernels, and fusing them roughly halves the step time on one GPU (measured
1.8-2.2x on RTX 4090/5090 in the sister port of this code). The maths is
unchanged fp32; results move only by floating-point rounding, which is below
the run-to-run nondeterminism of the atomics in the backward pass. The logged
PC2 runs did not use it (added 2026-09-24 for the local reproduction run,
`slurm/nmt_local.sbatch`). Checkpoints of a compiled run carry an
`_orig_mod.` prefix on the decoder keys.

**Basis implementation.** `hpc/env.sh` sets `RATIONAL_BASIS_IMPL=eager`, and
every logged NMT run used the eager evaluation. The fused Triton kernels in
`rational_cnn/triton_basis.py` are the package default on CUDA. They are
mathematically equivalent but associate the polynomial sums differently, so
set `RATIONAL_BASIS_IMPL=eager` outside `hpc/env.sh` as well to reproduce the
logs.

## Data

NMT reads the raw images and annotations directly and computes no features
up front. It uses the same dataset trees as the DGMC experiments (see the
repository README):

* **SPair-71k**: the extracted release, i.e. `data/SPair-71k` with
  `JPEGImages/`, `Layout/` and `PairAnnotation/` (`cfg.SPair.ROOT_DIR`).
* **PascalVOC-Keypoints**: the `raw/` tree that
  `../experiments/prepare_pascal_voc.py` downloads, i.e.
  `data/PascalVOC/raw/` with `images/` (VOC2011 `JPEGImages/` and
  `Annotations/`), `annotations/` (the Berkeley keypoints) and `splits.npz`
  (`cfg.VOC2011.ROOT_DIR`, `KPT_ANNO_DIR`, `SET_SPLIT`). `splits.npz` already
  has the layout `data/pascal_voc.py` expects, so ThinkMatch's
  `voc2011_pairs.npz` is not needed.

**On another machine** edit the absolute PC2 paths in three places:

* the base configs: `SPair.ROOT_DIR` (`spair_vgg16.json`), `VOC2011.*` and
  `CACHE_PATH` (the xml-list pickle, `voc_vgg16.json`), and `model_dir` in
  both. `BACKBONE_DIR` is read only by the `swin` backbone.
  `VOC2011.ROOT_DIR` must end in a slash, because `data/pascal_voc.py` builds
  paths by string concatenation.
* `hpc/env.sh`: `NMT_PFS_ROOT`, `NMT_GROUP_ROOT` / `NMT_CONDA_ENV`, and
  `NMT_SPAIR_ROOT` / `NMT_VOC_ROOT`. The Slurm scripts check the dataset
  trees there, and training reads the configs, so the two must point to the
  same trees.
* the `#SBATCH` account, partition and gres lines of `slurm/nmt_*.sbatch`.

## Running

The paper's 15 configurations per dataset are the default sweep plus `mv_K1`
and `mv_K2`, all with seeds 0-4:

```bash
slurm/submit.sh                        # SPair-71k: the 13-config default sweep x seeds 0-4 = 65 jobs
slurm/submit.sh mv_K1 mv_K2            # SPair-71k: the remaining 2 configs x seeds 0-4
DATASET=voc slurm/submit.sh            # the same two lines on PascalVOC
DATASET=voc slurm/submit.sh mv_K1 mv_K2
DRY_RUN=1 slurm/submit.sh              # write the configs, print the jobs, submit nothing
python slurm/make_config.py --list     # every configuration name; * marks the default sweep
```

`slurm/submit.sh` writes one config per (name, seed) into
`$NMT_RUN_ROOT/configs/` and submits one single-H100 job each
(`slurm/nmt_spair.sbatch`, `slurm/nmt_voc.sbatch`), so the exact settings of
every run are kept next to its outputs. Note that `DRY_RUN=1` writes those
configs too. Job logs go to `slurm/logs/`. Without Slurm, one run is

```bash
mkdir -p runs
export WANDB_MODE=offline RATIONAL_BASIS_IMPL=eager    # what the sbatch files set
python slurm/make_config.py mv_K4 0 runs/voc_mv_K4_seed0.json runs/voc_vgg16 voc
python -m torch.distributed.run --nproc_per_node=1 train_eval.py runs/voc_mv_K4_seed0.json
```

`train_eval.py` writes checkpoints and `settings.json` to `model_dir` and
per-epoch error dumps to `errors/` in the working directory. `runs/`,
`errors/` and `slurm/logs/` are git-ignored.

The configuration names are those of the DGMC sweep in `../slurm/configs.sh`,
so every NMT run has a DGMC counterpart. Only `cfg.SPLINE_CNN` differs between
them:

| config | basis | K | graph-conv params |
|---|---|---|---|
| `spline`, `spline_k3`, `spline_k2` | B-spline hats (the baseline) | 25, 9, 4 | 28.17 M, 10.84 M, 5.42 M |
| `rational`, `rational_k3` | product of 1-D rationals, hat init | 25, 9 | 28.17 M, 10.84 M |
| `rational_mv_k3` | multivariate rational, hat init | 9 | 10.84 M |
| `rational_mv_d54` | multivariate, degrees (5, 4) | 25 | 28.17 M |
| `mv_K1` ... `mv_K16` | multivariate, PCA + vp init | 1, 2, 4, 6, 9, 16 | 2.17-18.42 M |
| `mlp_K4`, `mlp_K9` | filter-generating MLP (control) | 4, 9 | 5.42 M, 10.84 M |

`rational_mv` (K = 25) is defined as well but was not run. Configurations with
equal K have equal parameter counts, so for example `mv_K4`, `spline_k2` and
`mlp_K4` (all 5.42 M) form a controlled comparison.

**Recipe.** All configurations keep NMT's recipe: `long_halving5` (LR 5e-4,
dropped 10x at epochs 2 and 5), 2000 iterations per epoch, 1000 evaluation
samples per class, and the backbone fine-tuned at 0.03x LR. The batch size is
8 on PascalVOC (NMT's own setting in `voc_basic.json`) and 5 on SPair-71k.
Images are resized to 256 x 256.

**Runs stop after 7 of the schedule's 12 epochs** (`cfg.TRAIN.max_epochs`,
where 0 means the whole schedule). The LR milestones are unchanged, so both
decays happen and the last two epochs run at the final rate. The mean
accuracy still rises by 0.0-0.2 points per epoch at the end, so the runs are
truncated, not converged, and the best test epoch is the last one in 122 of
the 150 runs. Evaluation and
checkpointing run every epoch. On one H100 a run takes 1.3 h on SPair-71k
(1.0-1.9 h) and 2.2 h on PascalVOC (1.2-3.4 h), averaged over the logged
jobs. The sbatch files ask for 18 h, a limit sized for the SwinV2 backbone.

**Outputs.** `train_eval.py` calls `wandb.init()` unconditionally with a
fixed W&B entity, so the jobs set `WANDB_MODE=offline`. `WANDB=disabled`
also works. `WANDB=online` works only after the `entity` argument in
`train_eval.py` is changed to your own. Model and Adam state take ~1.3 GB per
epoch, so the generated configs set `keep_last_checkpoint_only: true`, which
keeps only the newest epoch.

## Results

```bash
python results/analyze.py --dataset spair    # or --dataset voc; --no-params skips the parameter count
python results/analyze.py --dataset voc --logs slurm/logs    # fresh job logs instead
```

It reads the checked-in logs of the paper runs in
`../results/logs/nmt_spair/` and `../results/logs/nmt_voc/` (75 each,
15 configurations x 5 seeds). They are the Slurm job logs with every progress
bar collapsed to its final state, and they give output identical to the full
logs. One PascalVOC job (`spline_k3`, seed 0) aborted on a port clash before
its first epoch and was re-run, and only the re-run is kept. The script prints
mean ± std over seeds, the per-epoch curve, a per-class comparison and Welch
t-tests in the same layout as the DGMC `../results/analyze.py`.

Keypoint accuracy (%), mean over classes and ± std over 5 seeds, at the best
test epoch (the NMT columns of the paper's keypoint table). Final-epoch
accuracy differs from it by at most 0.04.

| config | graph conv | PascalVOC | SPair-71k |
|---|---|---|---|
| `spline` | 28.17 M | 83.10 ± 0.36 | 82.10 ± 0.11 |
| `spline_k3` | 10.84 M | 82.66 ± 0.30 | 81.54 ± 0.31 |
| `spline_k2` | 5.42 M | 80.10 ± 0.46 | 77.79 ± 0.40 |
| `rational` | 28.17 M | **83.78 ± 0.35** | 82.08 ± 0.21 |
| `rational_k3` | 10.84 M | 82.85 ± 0.42 | 81.64 ± 0.35 |
| `rational_mv_k3` | 10.84 M | 83.08 ± 0.26 | **82.11 ± 0.35** |
| `mv_K1` | 2.17 M | 79.27 ± 0.69 | 77.61 ± 0.19 |
| `mv_K2` | 3.25 M | 81.11 ± 0.78 | 80.20 ± 0.48 |
| `mv_K4` | 5.42 M | 82.42 ± 0.43 | 81.52 ± 0.22 |
| `mv_K6` | 7.59 M | 82.95 ± 0.42 | 81.71 ± 0.18 |
| `mv_K9` | 10.84 M | 82.88 ± 0.38 | 81.59 ± 0.14 |
| `mv_K16` | 18.42 M | 83.01 ± 0.33 | 81.69 ± 0.27 |
| `mlp_K4` | 5.42 M | 81.90 ± 0.27 | 80.40 ± 0.57 |
| `mlp_K9` | 10.84 M | 82.12 ± 0.51 | 81.05 ± 0.32 |
| `rational_mv_d54` | 28.17 M | 83.35 ± 0.38 | 82.01 ± 0.21 |

Two caveats when reading these next to the DGMC tables:

* NMT reports **keypoint-weighted** accuracy over 1000 randomly sampled test
  pairs per class. The DGMC SPair-71k headline is pair-averaged over all
  12,234 test pairs, and its keypoint-weighted column is the comparable one.
* The backbone differs from the NMT paper, so these numbers reproduce neither
  its SwinV2 results nor the DGMC ones. They compare bases within NMT at a
  DGMC-like backbone scale.

## Local 12-epoch runs (2026-09-25)

`../results/logs/nmt_local_12ep/` holds eleven runs made on the group's local
cluster (RTX 4090/5090, `slurm/nmt_local.sbatch`, `"compile": true`) with the
configuration of the sweep above but the **full 12-epoch** `long_halving5`
schedule (`TRAIN.max_epochs: 0`), i.e. epochs 1-7 replay the sweep's recipe
and epochs 8-12 show how much the 7-epoch budget leaves on the table. Seed 0
throughout, plus a second spline seed on SPair-71k. Keypoint accuracy (%):

| config | VOC ep 7 | VOC ep 12 | SPair ep 7 | SPair ep 12 |
|---|---|---|---|---|
| `spline` | 83.10 | 83.53 | 81.36 (seed 1: 82.26) | 81.46 (seed 1: 82.42) |
| `rational` | 83.47 | 83.64 | 82.21 | 82.33 |
| `rational_mv_k3` | 83.02 | 83.26 | 81.66 | 81.81 |
| `mv_K6` | 82.81 | 82.94 | 81.36 | 81.50 |
| `mv_K9` | 82.46 | 82.73 | 81.22 | 81.20 |

The epoch-7 values reproduce the PC2 means of the table above within 0.1-0.4
(the seed-0 SPair-71k `spline` run is an outlier; seed 1 is in the PC2 range).
The remaining five epochs add 0.1-0.4 points to every configuration and keep
the ordering, so the ceiling of this recipe is about 83.5-83.7 on PascalVOC
and 82.3-82.4 on SPair-71k. The logs are the run logs with the per-iteration
progress lines removed.

## Tests

```bash
python -m pytest        # from nmt/
```

The 11 tests run on CPU without a dataset (the VGG16 test loads the ImageNet
weights). They cover the VGG16 feature shapes and optimizer group, both
convolutions and their rejection of bad settings, the image-size coupling of
the pseudo-coordinates, and the SPair-71k file-name compatibility.

## Licence

`utils/swinV2.py` is Microsoft's Swin Transformer V2 (MIT, per its header).
`utils/timmLayers/` comes from `timm` by Ross Wightman (Apache-2.0). The
rotary embeddings in `model/nGPT_*.py` are adapted from
`lucidrains/rotary-embedding-torch`. `utils/vit.py` and `utils/gmt.py`
contain timm's `drop_path` and DINO-style positional-embedding
interpolation. Upstream NMT ships no licence file.
