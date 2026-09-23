# PC² Noctua 2 runbook

How this clone of RationalBasisCNN is wired into PC² Noctua 2, and how to run
the paper's experiments here.

## 1. Why anything had to change

`$HOME` (`/pc2/users/h/hpcabpo`) is quota-limited. Upstream, the repository
writes several GB into it:

| upstream location | what lands there |
|---|---|
| `<repo>/data/` | PascalVOC-Keypoints (~2 GB raw + processed) and FAUST |
| `<repo>/wandb/` | Weights & Biases run directories |
| `~/.cache/torch/` | torchvision's VGG16 weights (~550 MB) |
| `~/.cache/pip`, `~/.conda` | package and environment caches |
| `~/.nv/ComputeCache` | CUDA JIT cache |

`hpc/env.sh` redirects every one of these onto project storage. Nothing is
symlinked and no path is hardcoded in the experiment scripts: the four
`--root` defaults read `$RBCNN_DATA_ROOT` and fall back to the upstream
`<repo>/data` when the variable is unset, so the repository still behaves
exactly as upstream on any other machine.

## 2. Paths

```bash
cd /pc2/users/h/hpcabpo/research/RationalBasisCNN
source hpc/env.sh
```

Account `hpc-prf-llmrout`. Two storage tiers, following the convention of
`projects/kaggle-RSNA-Knee-Abnormality-Detection`:

- **`/scratch/hpc-prf-llmrout/hpcabpo/rational-basis-cnn`** — bulk, sequential
  data: `data/` (the dataset root), `downloads/`, `cache/`, `wandb/`, `tmp/`.
- **`/pc2/groups/hpc-prf-llmrout/hpcabpo/rational-basis-cnn`** — the conda
  environment and any dataset made of very many tiny files, which PC²
  recommends keeping on group storage because of the small-file metadata load.

**`/pc2/groups` is mounted READ-ONLY on compute nodes** (`ro,relatime,vers=4.0`),
and read-write only on login nodes. This is the reason the two tiers exist and
why everything cannot simply live on group storage:

| | `/scratch` (GPFS) | `/pc2/groups` (NFS4) |
|---|---|---|
| compute nodes | read-write | **read-only** |
| login nodes | read-write | read-write |
| seq read, compute node | 5.2 GB/s | n/a (ro) |
| 3000 small file creates | <1 s | ~7 s (login node) |
| space / inodes | 2 TB / **1,000,000** | 1 TB / 2.1 billion |

So anything a *job* writes — checkpoints, logs, wandb, PyG `processed/`
directories, caches, tmp — must be on `/scratch`. Group storage holds immutable,
read-only artifacts: the conda environment and the SPair-71k raw data. Both were
verified readable from a compute node.

**How the read-only root is handled.** Datasets live on group storage, so all
of `$RBCNN_DATA_ROOT` is read-only inside jobs. That works for training because
PyG skips both download and processing when the raw and processed files already
exist (`Dataset._download` returns early if `raw_paths` exist,
`Dataset._process` if `processed_paths` do), so a prepared dataset is only read.
Preparation is the part that writes, and it also needs a GPU, so it cannot run
on a login node either. The split is therefore:

```text
prep job (compute node, GPU, writes)   ->  $RBCNN_DATA_STAGE   on /scratch
hpc/promote_dataset.sh (login node)    ->  $RBCNN_DATA_ROOT    on /pc2/groups
training jobs (compute node, reads)    <-  $RBCNN_DATA_ROOT
```

Both training runners fail fast if a dataset was never promoted, rather than
dying with an opaque "Read-only file system" error inside the data loader:

```text
ERROR: PascalVOC is not prepared at /pc2/groups/.../data
  1) sbatch slurm/prep_pascal_voc.sbatch
  2) bash hpc/promote_dataset.sh PascalVOC   # from a login node
```

SPair-71k follows the same rule: `slurm/prep_spair71k.sbatch` reads the raw
tree from group storage (reading is fine inside a job), writes the processed
graphs to `$RBCNN_DATA_STAGE/SPair71k`, and `bash hpc/promote_dataset.sh
SPair71k` moves them next to the raw tree.

**The `/scratch` inode quota is the second binding constraint, not disk space.**
The fileset allows 1,000,000 inodes and is already at ~93 % — `rsna-knee/competition`
alone holds ~848,000 of them (DICOM slices). Note that `df` reports the fileset
quota only when the path you give it is *inside* the fileset: `df -i $RBCNN_PFS_ROOT`
shows the 1,000,000 limit, while `df -i /scratch` shows the whole 4.2-billion-inode
filesystem and is misleading. The same is true of `df -h ~` versus `df -h /pc2/users`
for the 50 GB home quota. Group storage has 2.1 billion inodes at 1 % used, so
small-file datasets go there.

What `hpc/env.sh` exports:

| variable | purpose |
|---|---|
| `RBCNN_DATA_ROOT` | canonical dataset root on **group storage**; holds `PascalVOC/`, `SPair-71k/`, `FAUST/`. Read-only in jobs. The `--root` default of every experiment script |
| `RBCNN_DATA_STAGE` | writable staging root on `/scratch`, used by the prep jobs before promotion |
| `RBCNN_SPAIR_ROOT` | `$RBCNN_DATA_ROOT/SPair-71k` |
| `RBCNN_DOWNLOAD_ROOT` | staging area for manually obtained archives (`MPI-FAUST.zip`) |
| `RBCNN_CACHE_ROOT` | parent of every cache below |
| `TORCH_HOME` | torchvision VGG16 weights |
| `XDG_CACHE_HOME`, `PIP_CACHE_DIR`, `MPLCONFIGDIR` | generic caches |
| `PYTHONPYCACHEPREFIX` | bytecode, keeps `__pycache__` out of the repo |
| `CUDA_CACHE_PATH`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` | GPU JIT caches |
| `CONDA_ENVS_PATH`, `CONDA_PKGS_DIRS`, `RBCNN_CONDA_ENV` | conda on group storage |
| `WANDB_DIR`, `WANDB_CACHE_DIR`, `WANDB_ARTIFACT_DIR` | wandb run data |
| `TMPDIR` | login-node scratch; Slurm jobs keep their node-local `/tmp` |

Small text logs deliberately stay in the repository: `slurm/logs/` (gitignored
stdout/stderr) and `results/logs/` (the training logs `results/analyze.py`
reads). See the warning in §5.

## 3. Environment

PyG's C++ extensions (`torch-spline-conv`, `pyg-lib`, `torch-scatter`) are
**not** needed — `rational_cnn/bspline.py` is a pure-PyTorch SplineConv.

`environment-hpc.yml` pins the stack the README reports as tested, torch 2.12
/ PyG 2.8 / CUDA 13. The H100 nodes run driver 595.71.05, which supports the
CUDA 13 wheels.

```bash
bash hpc/create_environment.sh
```

Creates or updates `$RBCNN_CONDA_ENV`, installs the package editable, and runs
the unit suite. It submits no Slurm job.

## 4. Data

**PascalVOC-Keypoints** downloads automatically. Compute nodes on this
cluster do have outbound internet (verified against both the Internet Archive
annotation mirror and `download.pytorch.org`), so the prep job downloads and
extracts VGG16 features in one GPU allocation:

```bash
sbatch slurm/prep_pascal_voc.sbatch          # GPU job -> $RBCNN_DATA_STAGE/PascalVOC
bash hpc/promote_dataset.sh PascalVOC        # login node -> $RBCNN_DATA_ROOT
rm -rf "$RBCNN_DATA_STAGE/PascalVOC"         # reclaim ~65,000 /scratch inodes
```

It took 1 m 44 s on an H100, not the ~20 min the README estimates for an
RTX 4090. The result is 20 categories, 6953 train / 1671 test graphs with
1024-d VGG16 features.

**FAUST** cannot be downloaded: it must be requested from
http://faust.is.tue.mpg.de/ and its license forbids redistribution. Once you
have the archive, stage it and process it:

```bash
cp /path/to/MPI-FAUST.zip "$RBCNN_DOWNLOAD_ROOT/"
python experiments/prepare_faust.py --src "$RBCNN_DOWNLOAD_ROOT/MPI-FAUST.zip" \
       --root "$RBCNN_DATA_STAGE"
bash hpc/promote_dataset.sh FAUST
```

**SPair-71k** (Min et al., ICCV 2019 — 18 categories, 1,800 images, 70,958
annotated pairs) with its keypoint annotations:

```bash
bash hpc/download_spair71k.sh      # idempotent; verifies size + SHA-256 + entry count
```

Two things about this one are not obvious:

- The official host `cvlab.postech.ac.kr` is **unreachable** — DNS resolves to
  141.223.85.126 but ports 80 and 443 both time out, while every other external
  host works. The script therefore pulls the Internet Archive's 2025-07-01
  snapshot of the original tarball, the same fallback
  `experiments/prepare_pascal_voc.py` already uses for the PascalVOC keypoint
  annotations. The snapshot's `x-archive-orig-content-length` equals its
  `content-length`, so it is the complete original file
  (226,961,117 bytes, SHA-256 `d145eebe…08809`). The `0jl/SPair-71k`
  HuggingFace dataset is *not* a mirror — it only contains a loader script
  pointing at the same dead URL.
- It is **76,438 mostly-tiny JSON files**, which does not fit in the remaining
  `/scratch` inode budget. It lives on group storage at `$RBCNN_SPAIR_ROOT` and
  is symlinked to `$RBCNN_DATA_ROOT/SPair-71k`, so it is still reachable at the
  conventional data root.

Contents: `ImageAnnotation/<category>/<image>.json` carries per-image keypoints
(`kps`), bounding box, pose, azimuth and the occluded/truncated/difficult flags;
`PairAnnotation/{trn,val,test}/` carries the 53,340 / 5,384 / 12,234 pairs with
matched `src_kps`/`trg_kps` and SPair's difficulty factors (viewpoint and scale
variation, truncation, occlusion). All 1,800 image annotations have non-empty
keypoints. `experiments/spair71k.py` runs the benchmark; see §5.

Then build the SPair-71k graphs (VGG16 features for all 1,800 images and the
pair tables of both layouts) and promote them:

```bash
sbatch slurm/prep_spair71k.sbatch            # 1 H100, reads $RBCNN_SPAIR_ROOT
bash hpc/promote_dataset.sh SPair71k         # login node -> $RBCNN_DATA_ROOT/SPair71k
rm -rf "$RBCNN_DATA_STAGE/SPair71k"          # a handful of files; optional
```

The result is `$RBCNN_DATA_ROOT/SPair71k/processed/{images,pairs}.pt`. The
dataset class defines no `download()`, so PyG never looks at the raw tree once
those two files exist -- training only ever reads them.

`--root` defaults to `$RBCNN_DATA_ROOT` for all of these; pass it explicitly
to override.

## 5. Running the experiments

```bash
sbatch slurm/prep_pascal_voc.sbatch   # once, then promote (see Data above)
sbatch slurm/prep_spair71k.sbatch     # once, then promote (see Data above)
slurm/submit.sh voc                   # paper configs x seeds 0-4
slurm/submit.sh spair                 # same configs on SPair-71k x seeds 0-4
DRY_RUN=1 slurm/submit.sh spair       # list the 65 jobs without submitting
slurm/submit.sh faust                 # paper configs x seeds 0-2
slurm/submit.sh voc mv_K4 mv_K6       # a subset
SEEDS="5 6" WANDB=1 slurm/submit.sh faust faust_mv_K8
python results/analyze.py             # tables + Welch t-tests
```

Jobs request `--partition=gpu_h100 --gres=gpu:h100:1 --account=hpc-prf-llmrout`,
one job per (config, seed): 6 h for VOC, 12 h for FAUST. `gpu_h100` is
routinely fully subscribed, so short wall times backfill far better — the prep
job was cut from 6 h to 2 h for that reason. `gpu_a40` is a much less
contended alternative for these small graph-conv runs.

`PYTHON` defaults to `$RBCNN_CONDA_ENV/bin/python`; override it to use a
different interpreter.

### SPair-71k

One job = one configuration and seed on one H100
(`slurm/spair71k.sbatch`: 8 CPUs, 48 GB, 2 h). A single run:

```bash
CONFIG=mv_K4 SEED=0 sbatch slurm/spair71k.sbatch
# or equivalently
SEEDS=0 slurm/submit.sh spair mv_K4
```

Knobs, passed through the environment: `EPOCHS` (15), `LAYOUT` (`large`;
`small` logs to `results/logs/spair71k_small/`), `NUM_WORKERS` (6, graphs are
built on the fly), `WANDB=1`. `slurm/submit.sh spair` refuses to submit
anything until the graphs are promoted, and every job re-checks that at start.

**The 2 h wall time is an estimate, not a measurement.** PascalVOC trains
~1,300 pairs/s on an H100, which puts SPair-71k at roughly a minute per epoch
(53,340 training + 17,618 evaluation pairs), ~15-20 min per run. Submit one
job first, read `Time:`/`Eval:` in its log, then tighten `#SBATCH --time`.
The full default sweep is 13 configs × 5 seeds = **65 H100 jobs**.

Protocol, as implemented in `rational_cnn/spair.py`:

- the fixed pairs of the `large` layout, no difficulty filtering, only the
  keypoints visible in both images (pygmtools' `intersection` filter; a pair's
  `kps_ids` equals that set for all 70,958 pairs);
- target keypoints shuffled -- randomly in training, one fixed permutation per
  pair in evaluation, identical across configs and seeds. Without it the
  ground truth is the identity, because `kps_ids` is sorted in both images;
- node features exactly as PyG's PascalVOCKeypoints, so the backbones see the
  same input distribution as in Experiment 1.

Reported per category and as the mean over 18 categories, on val and on all
12,234 test pairs after every epoch: pair-averaged accuracy (headline, the
pygmtools convention) and keypoint-weighted accuracy (the `pascal_voc.py`
convention). `results/analyze.py` tabulates final test accuracy, test accuracy
at the best validation epoch, best test accuracy (optimistic, for comparison
with the PascalVOC `best acc` column) and final keypoint accuracy.

Two caveats when comparing with published SPair-71k numbers: the crop keeps
PyG's 16 px margin around all keypoints, whereas pygmtools rescales keypoints
to the raw bounding box; and DGMC here uses frozen VGG16 features, whereas
e.g. BBGM and NGMv2 fine-tune the CNN end to end. The numbers are meant for
comparing bases against each other, not against those leaderboards.

> **Warning.** `slurm/{pascal_voc,faust}.sbatch` write
> `results/logs/<task>/<config>_seed<N>.log`, which is exactly where the
> paper's 137 checked-in logs live. Re-running a paper config with the same
> seed **overwrites** the upstream log. They are tracked by git, so
> `git checkout -- results/logs` restores them. To keep reproduction runs
> separate, export a different root before submitting:
>
> ```bash
> export RBCNN_RESULTS_LOG_ROOT="$RBCNN_PFS_ROOT/results-logs"
> ```
>
> `results/analyze.py` reads the in-repo `results/logs/` by default; point it
> at the redirected root with `python results/analyze.py --logs "$RBCNN_RESULTS_LOG_ROOT"`.

### Regional (domain-split) rational basis

Tests whether confining the learnable rational basis to regions of the
pseudo-coordinate domain helps, on SPair-71k with DGMC.

```bash
slurm/submit.sh spair zone_K4 zone_K9 mv_K4_frozen mv_K9_frozen   # 20 jobs, ~3 min each
python results/analyze.py                                          # read the SPair table
```

**The basis.** `B_{m,p}(u) = w_m(u) * R_{m,p}(u)`: the domain is split into
`zones` rings around its centre, each holding `bases_per_zone` ordinary
safe-Pade rationals, so `K = zones * bases_per_zone`. The windows `w_m` are
Wendland bumps in `rho = ||u - 0.5||`, Shepard-normalised so they sum to 1.
Two properties, both pinned by `tests/test_regional.py`:

* each basis function is **exactly** zero outside its ring (measured max
  |B| outside = 0.0, even with coefficients pushed far from init) -- a plain
  rational can never vanish on an open set, which is why a compactly supported
  window is needed;
* the pieces merge into one continuous function with nothing to solve, because
  the windows are already a partition of unity.

**Rings, not a grid.** A grid of `k` splits per axis gives `k^D` cells -- 9 in
2-D, 27 in 3-D -- which would re-lock `K` to the kernel grid, the very thing
`--num_bases` exists to avoid. Radial zones give exactly `zones` regions in any
dimension (verified for D=2 and D=3). The default ring boundaries are the
equal-mass quantiles measured on 127,274 SPair training edges
(`DEFAULT_ZONE_RADII`): each zone then holds the same share of real edges
(1.0x spread), where a 3x3 grid is 12x unbalanced.

**Matched comparators**, so only the basis differs:

| config | K | total params | compare against |
|---|---|---|---|
| `zone_K4` (4 zones x 1) | 4 | 1,935,873 | `mv_K4` 1,935,873, `spline_k2`, `mlp_K4` |
| `zone_K9` (3 zones x 3) | 9 | 3,739,553 | `mv_K9` / `rational_mv_k3` 3,739,553, `spline_k3`, `mlp_K9` |

`zone_K4` and `mv_K4` agree to the parameter, as do `zone_K9` and `mv_K9`.

**The `--freeze_basis` control.** `mv_K4_frozen` / `mv_K9_frozen` hold the basis
shapes at their initialisation and train only the kernel weights. The existing
evidence confounds two variables -- `spline` is fixed *and* local, `mv` is
learned *and* global -- and this supplies the missing fixed+global cell, so the
sweep can tell "locality helps" from "not learning the shape helps".

**Read the results with model selection.** 57 of 65 SPair runs peak at epoch 1
of 15, because a SPair epoch is 7.7x the updates of a PascalVOC epoch. Compare
the `@best val` column of `results/analyze.py`, not `final acc`, and use paired
tests (seed fixes init and data order across configs).

**What would falsify it.** `zone_K4` ~ `mv_K4` at matched parameters means
locality does nothing here. If instead `mv_K4_frozen` alone recovers most of
the gap, the operative variable was fixedness, not support.

### HPatches transfer (out-of-task)

Does a keypoint refinement trained for sparse *semantic* matching help
*geometric* matching, frozen? The modules come from the NMT runs, so nothing is
retrained here.

```bash
bash hpc/download_hpatches.sh            # 1.2 GiB -> group storage, 1463 files
slurm/submit_hpatches.sh                 # one job per configuration, its 5 seeds inside
DRY_RUN=1 slurm/submit_hpatches.sh       # print the jobs, submit nothing
slurm/submit_hpatches.sh spline mv_K4    # a subset
python results/analyze_hpatches.py       # -> $RBCNN_PFS_ROOT/hpatches/k128_d1.0
# one run, for a quick look:
python experiments/hpatches_transfer.py \
  --run_dir /scratch/hpc-prf-llmrout/hpcabpo/nmt/runs/spair_vgg16/spline_seed0
```

**Protocol.** N keypoints are detected in the reference image and mapped into
the target with the ground-truth homography, so the correct correspondence is
known by construction. Distractors, detected independently in the target and
kept at least 6 px from any true correspondence, are mixed in and the candidate
list is shuffled. A cosine matcher picks one of the N + M candidates. Only how a
descriptor is produced differs between arms:

| arm | what it isolates |
|---|---|
| `raw` | the frozen VGG16 descriptor (baseline) |
| `trained` | NMT's refinement `psi(x, G) + Wx` with its trained weights |
| `random` | same architecture untrained: learned prior vs operator shape |
| `mean` | mean of the Delaunay neighbours: learned prior vs plain smoothing |
| `geometry` | the trained module fed **constant** features: appearance-free |
| `geometry_random` | as above, untrained |

**Two traps, both worth knowing.**

*The distractors are not optional.* Without them the candidates are exactly the
homography image of the queries, the two Delaunay graphs are near-isomorphic,
and matching succeeds on topology alone: the appearance-free `geometry` arm
scored 88.1 overall and 76.5 on viewpoint pairs, beating the trained module fed
real descriptors. `--distractors 0` reproduces that leaky setting. With one
distractor per query the shortcut falls to 33.6.

*Only the 59 `v_` sequences carry information.* HPatches illumination sequences
have **identity** homographies, so the keypoint sets coincide and even untrained
weights with constant features score 99.98. `results/analyze_hpatches.py`
therefore defaults to the viewpoint split.

**First result** (spline_seed0, one seed, 128 queries + 128 distractors,
viewpoint sequences, chance 0.44 %):

| arm | accuracy |
|---|---|
| raw | 42.08 |
| trained | **43.98** |
| random | 41.11 |
| mean | 41.88 |
| geometry | 33.57 |
| geometry_random | 12.96 |

The refinement transfers: +1.9 over raw descriptors and +2.9 over the same
architecture untrained. What transfers is mostly geometric -- the
appearance-free arm still reaches 33.6 against 13.0 untrained -- so it is a
*learned* geometric prior, not the operator's shape. On illumination pairs the
refinement *hurts* (68.6 vs 78.5 raw): with the viewpoint fixed, raw VGG
features are already strong and mixing in neighbours blurs them.

Caveat: 128 keypoints per image is far outside the 7-20 node graphs these
modules trained on. Sweep `--num_keypoints 32 64 128` before concluding.

## 6. Monitoring

```bash
squeue --me
sacct -j JOB_ID --format=JobID,JobName%24,State,Elapsed,AllocTRES,ExitCode
tail -f slurm/logs/voc-JOBID.out
```

## 7. Checking the quota is actually safe

After a run, nothing new should have appeared in `$HOME`:

```bash
du -sh ~/.cache ~/.nv ~/.local ~/.conda
du -sh "$RBCNN_PFS_ROOT"/* "$RBCNN_GROUP_ROOT"/*
df -i "$RBCNN_PFS_ROOT"          # inodes, the scarce resource on /scratch
du --inodes -d 2 /scratch/hpc-prf-llmrout/hpcabpo | sort -rn | head
```

Note that `~/.cache` (3.2 GB) and `~/.local` (1.8 GB) were already populated
by earlier work before this project was set up; this project adds to neither.
