r"""Writes one NMT experiment config for a (configuration, seed) pair.

The configuration names are exactly those of the DGMC sweep in
`../slurm/configs.sh`, so every NMT run has a DGMC counterpart to compare with.
Only the basis of the continuous-kernel convolution differs between them;
everything else comes from the per-dataset base file in BASES below.

That base file is inlined rather than chained, because utils/utils.py resolves
only two levels of "default_json" (the generated file plus
experiments/voc_basic.json).

Usage:
    python slurm/make_config.py --list
    python slurm/make_config.py <name> <seed> <out.json> <run_root> [dataset]

`dataset` is 'spair' (default) or 'voc'. The same CONFIGS table serves both, so
a name means the same basis in either dataset and the two NMT result columns
line up row by row.
"""
import json
import os.path as osp
import sys

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))

# dataset -> the experiment json the generated config inherits from. Both files
# pin the same VGG16 backbone, IMAGE_SIZE and 7-epoch budget; they differ only
# in the dataset paths and in the batch size each host recipe prescribes.
BASES = {
    'spair': osp.join(ROOT, 'experiments', 'spair_vgg16.json'),
    'voc':   osp.join(ROOT, 'experiments', 'voc_vgg16.json'),
}
DEFAULT_DATASET = 'spair'

_MV = dict(conv='rational', rational_basis='multivariate', degrees=[8, 6])
_PCA = dict(_MV, init='pca', vp=True)

# name -> cfg.SPLINE_CNN overrides. Kernel size, degrees, init and so on that
# are not named here fall back to experiments/spair_vgg16.json and then to the
# defaults in utils/config.py, exactly as the DGMC flags do.
CONFIGS = {
    # B-spline hats: what NMT used originally (kernel_size 5 => K = 25)
    'spline':          dict(conv='spline', kernel_size=5),
    'spline_k3':       dict(conv='spline', kernel_size=3),   # K = 9
    'spline_k2':       dict(conv='spline', kernel_size=2),   # K = 4
    # tensor product of 1-D rational functions, fitted to the hats
    'rational':        dict(conv='rational', rational_basis='product', kernel_size=5),
    'rational_k3':     dict(conv='rational', rational_basis='product', kernel_size=3),
    # non-separable multivariate rational functions
    'rational_mv':     dict(_MV, kernel_size=5),
    'rational_mv_k3':  dict(_MV, kernel_size=3),
    'rational_mv_d54': dict(conv='rational', rational_basis='multivariate',
                            degrees=[5, 4], kernel_size=5),
    # free number of basis functions: PCA of the hats + variance-preserving init
    'mv_K1':  dict(_PCA, num_bases=1),
    'mv_K2':  dict(_PCA, num_bases=2),
    'mv_K4':  dict(_PCA, num_bases=4),
    'mv_K6':  dict(_PCA, num_bases=6),
    'mv_K9':  dict(_PCA, num_bases=9),
    'mv_K16': dict(_PCA, num_bases=16),
    # filter-generating MLP instead of a rational basis (control)
    'mlp_K4': dict(conv='rational', rational_basis='mlp', vp=True, num_bases=4),
    'mlp_K9': dict(conv='rational', rational_basis='mlp', vp=True, num_bases=9),
}

# The sweep slurm/submit.sh runs by default: the same 13 configurations as
# the DGMC sweep `../slurm/submit.sh spair` at the repository root.
DEFAULT_SWEEP = ['spline', 'spline_k3', 'spline_k2', 'rational', 'rational_k3',
                 'rational_mv_k3', 'mv_K4', 'mv_K6', 'mv_K9', 'mv_K16',
                 'mlp_K4', 'mlp_K9', 'rational_mv_d54']


def make(name, seed, run_root, dataset=DEFAULT_DATASET):
    if name not in CONFIGS:
        raise SystemExit(f"unknown configuration {name!r}; "
                         f"one of: {' '.join(sorted(CONFIGS))}")
    if dataset not in BASES:
        raise SystemExit(f"unknown dataset {dataset!r}; "
                         f"one of: {' '.join(sorted(BASES))}")
    with open(BASES[dataset]) as f:
        cfg = json.load(f)
    cfg.pop('_comment_backbone', None)
    cfg['SPLINE_CNN'] = {**cfg.get('SPLINE_CNN', {}), **CONFIGS[name]}
    # NMT trains about 11 epochs per day, so the sweep is truncated to 7.
    # The LR milestones (2, 5) are untouched, so both decays still happen and
    # the last two epochs run at the final learning rate.
    cfg['TRAIN'] = {**cfg.get('TRAIN', {}), 'max_epochs': 7}
    cfg['RANDOM_SEED'] = int(seed)
    cfg['model_dir'] = osp.join(run_root, f'{name}_seed{seed}')
    # A 13 x 5 sweep keeping every epoch would write about 1 TB.
    cfg['keep_last_checkpoint_only'] = True
    cfg['_generated_by'] = f'slurm/make_config.py {name} {seed} ({dataset})'
    return cfg


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--list':
        for name in CONFIGS:
            mark = '*' if name in DEFAULT_SWEEP else ' '
            print(f'{mark} {name}')
        raise SystemExit(0)
    if len(sys.argv) not in (5, 6):
        raise SystemExit(__doc__)
    name, seed, out, run_root = sys.argv[1:5]
    dataset = sys.argv[5] if len(sys.argv) == 6 else DEFAULT_DATASET
    with open(out, 'w') as f:
        json.dump(make(name, seed, run_root, dataset), f, indent=2)
    print(out)
