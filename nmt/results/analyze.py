r"""Aggregates the NMT job logs into the result table, mirroring the DGMC
results/analyze.py at the repository root: mean +- std over seeds, the
per-epoch curve, a per-class comparison and Welch t-tests between
configurations.

Every evaluation epoch prints a "Matching accuracy" block (the dataset's
classes plus their average) to the job's stdout, so the job logs are the
source of truth; nothing is read from the run directories.

The logs of the paper's runs are checked in at ../results/logs/nmt_spair/ and
../results/logs/nmt_voc/ as <config>_seed<N>.log, with their progress bars
collapsed to the final state, and --dataset reads the matching directory by
default. Fresh job logs in slurm/logs keep their Slurm names: slurm/submit.sh
writes nmt-spair-<config>-s<seed>-<job>.out for SPair-71k and
nmt-voc-<config>-s<seed>-<job>.out for PascalVOC, so --dataset also picks which
sweep to aggregate there. Mixing the two in one table would be wrong -- the
class sets and the metrics differ.

    python results/analyze.py [--dataset spair|voc] [--logs slurm/logs]
                              [--no-params]
"""
import argparse
import glob
import os.path as osp
import re
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
# The repository root, whose results/logs/ holds the checked-in logs.
REPO = osp.dirname(ROOT)

# A checked-in log, one directory per dataset: <config>_seed<N>.log
CHECKED_IN = r'(.+)_seed(\d+)\.log'

# Same display order as the DGMC table, so the two can be read side by side.
ORDER = ['spline', 'spline_k3', 'spline_k2', 'rational', 'rational_k3',
         'rational_mv', 'rational_mv_k3', 'mv_K1', 'mv_K2', 'mv_K4', 'mv_K6',
         'mv_K9', 'mv_K16', 'mlp_K4', 'mlp_K9', 'rational_mv_d54']

TESTS = [('rational_mv_k3', 'spline'), ('rational_mv_k3', 'rational_k3'),
         ('rational_mv_k3', 'spline_k3'), ('mv_K4', 'spline'),
         ('mv_K4', 'mlp_K4'), ('mv_K4', 'spline_k2'), ('mv_K9', 'mlp_K9'),
         ('mv_K9', 'rational_mv_k3'), ('mv_K16', 'mv_K9'),
         ('rational', 'spline'), ('rational_k3', 'spline_k3'),
         ('rational_mv_d54', 'rational_mv')]


# dataset -> (accepted job-name prefixes after "nmt-", number of classes,
# table title). SPair has two spellings: submit.sh gained a dataset segment
# when the second sweep was added, so the original 65 runs are nmt-<cfg>-s<n>
# and everything submitted since is nmt-spair-<cfg>-s<n>.
DATASETS = {
    'spair': (('spair-', ''), 18, 'NMT on SPair-71k, VGG16 backbone '
                                  '(keypoint accuracy, mean over 18 classes)'),
    'voc':   (('voc-',), 20, 'NMT on PascalVOC-Keypoints, VGG16 backbone '
                             '(keypoint accuracy, mean over 20 classes)'),
}


def log_pattern(dataset):
    r"""Regex matching only :obj:`dataset`'s job logs.

    The prefixes are tried longest first, so "nmt-spair-mv_K1-s0" consumes the
    prefix instead of yielding a configuration named "spair-mv_K1"; the
    lookahead then stops SPair's empty prefix from swallowing another
    dataset's logs."""
    own = sorted(DATASETS[dataset][0], key=len, reverse=True)
    others = sorted({p for d, (pfxs, _, _) in DATASETS.items() if d != dataset
                     for p in pfxs if p}, key=len, reverse=True)
    guard = (r'(?!' + '|'.join(re.escape(p) for p in others) + r')'
             if others else '')
    return (r'nmt-(?:' + '|'.join(re.escape(p) for p in own) + r')' + guard +
            r'(.+)-s(\d+)-(\d+)\.out')


def sd(x):
    return x.std(ddof=1) if len(x) > 1 else float('nan')


def parse_log(path):
    r"""Returns the per-epoch class accuracies of one run as a
    :obj:`[num_epochs, 19]` array (18 classes then their average)."""
    rows, classes, block = [], None, None
    for line in open(path, errors='ignore'):
        line = line.strip()
        if line == 'Matching accuracy':
            block = []
            continue
        if block is None:
            continue
        # "average = ..." closes the block and would otherwise be read as
        # just another class, so it has to be matched first.
        m = re.fullmatch(r'average = ([\d.]+)', line)
        if not m:
            m2 = re.fullmatch(r'([a-z]+) = ([\d.]+)', line)
            if m2:
                block.append((m2.group(1), float(m2.group(2))))
                continue
        if m:
            names = [c for c, _ in block]
            if classes is None:
                classes = names
            elif names != classes:
                raise ValueError(f'{path}: inconsistent class order')
            rows.append([v for _, v in block] + [float(m.group(1))])
        block = None
    width = (len(classes) + 1) if classes else 1
    return (np.array(rows) * 100 if rows else np.zeros((0, width))), classes


def graph_conv_params(name):
    r"""Parameters of the two-layer graph convolution for a configuration,
    built exactly as slurm/make_config.py configures it."""
    sys.path.insert(0, osp.join(ROOT, 'slurm'))
    sys.path.insert(0, ROOT)
    import make_config
    from utils.config import cfg
    import model.sconv_archs as sconv

    cfg.SPLINE_CNN.update(dict(conv='spline', kernel_size=5, dim=2, aggr='max',
                               pyg_init=True, rational_basis='product',
                               degrees=[5, 4], safe='B', poly='chebyshev',
                               init='spline', init_noise=1e-3, vp=False,
                               num_bases=0))
    cfg.SPLINE_CNN.update(make_config.CONFIGS[name])
    net = sconv.SConv(1024, 648)
    return sum(p.numel() for p in net.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='spair', choices=sorted(DATASETS),
                        help='which sweep to aggregate (default: spair)')
    parser.add_argument('--logs', default=None,
                        help='directory with <config>_seed<N>.log or '
                        'nmt-<config>-s<seed>-*.out job logs (default: '
                        '../results/logs/nmt_<dataset>, the paper runs)')
    parser.add_argument('--no-params', action='store_true',
                        help='skip building the graph convolutions to count '
                        'their parameters')
    args = parser.parse_args()
    if args.logs is None:
        args.logs = osp.join(REPO, 'results', 'logs', f'nmt_{args.dataset}')
    _, n_classes, title = DATASETS[args.dataset]
    pattern = log_pattern(args.dataset)

    runs, partial, classes = defaultdict(list), defaultdict(list), None
    for path in sorted(glob.glob(osp.join(args.logs, '*'))):
        name = osp.basename(path)
        m = re.fullmatch(pattern, name) or re.fullmatch(CHECKED_IN, name)
        if not m:
            continue
        cfg_name, seed = m.group(1), int(m.group(2))
        accs, names = parse_log(path)
        if names is not None:
            classes = names
        if len(accs) == 0:
            partial[cfg_name].append(f'seed{seed}@0ep')
            continue
        runs[cfg_name].append(dict(seed=seed, curve=accs[:, -1],
                                   final=accs[-1], best=accs[:, -1].max(),
                                   epochs=len(accs)))

    # A run still in the queue has evaluated only its first few epochs, and NMT
    # accuracy is still climbing there, so averaging it into its configuration
    # would drag that row down by whole points. Every configuration in a sweep
    # gets the same budget, so the longest run in the sweep defines it and
    # anything short of it is set aside rather than silently mixed in.
    budget = max((r['epochs'] for rs in runs.values() for r in rs), default=0)
    for cfg_name, rs in list(runs.items()):
        short = [r for r in rs if r['epochs'] < budget]
        if not short:
            continue
        runs[cfg_name] = [r for r in rs if r['epochs'] == budget]
        partial[cfg_name] += [f"seed{r['seed']}@{r['epochs']}ep" for r in short]
        if not runs[cfg_name]:
            del runs[cfg_name]

    print('=' * 84)
    print(title)
    if not runs:
        print(f'no {args.dataset} NMT logs found in {args.logs}')
        return
    if classes is not None and len(classes) != n_classes:
        print(f'WARNING: expected {n_classes} classes, logs have '
              f'{len(classes)}: {classes}')
    epochs = sorted({r['epochs'] for rs in runs.values() for r in rs})
    print(f'{sum(len(v) for v in runs.values())} runs, '
          f'{len(runs)} configurations, epochs per run: '
          f'{epochs[0] if len(epochs) == 1 else epochs}')
    if partial:
        print('runs without an evaluation (excluded):', dict(partial))

    cfgs = [c for c in ORDER if runs[c]] + sorted(set(runs) - set(ORDER))
    params = {}
    if not args.no_params:
        for c in cfgs:
            try:
                params[c] = graph_conv_params(c)
            except Exception:
                params[c] = None

    print()
    head = f"{'config':18s} {'graph conv':>11s} {'seeds':>5s} {'final acc':>15s} {'best acc':>15s}"
    print(head)
    for c in cfgs:
        rs = runs[c]
        fin = np.array([r['final'][-1] for r in rs])
        best = np.array([r['best'] for r in rs])
        p = params.get(c)
        p_str = f'{p/1e6:8.2f} M' if p else '        -'
        print(f'{c:18s} {p_str:>11s} {len(rs):5d} '
              f'{fin.mean():8.2f} ± {sd(fin):4.2f} {best.mean():8.2f} ± {sd(best):4.2f}')

    print('\nper-seed final acc:')
    for c in cfgs:
        rs = sorted(runs[c], key=lambda r: r['seed'])
        print(f'{c:18s}', ' '.join(f"{r['final'][-1]:6.2f}" for r in rs))

    print('\nacc per epoch (mean over seeds):')
    for c in cfgs:
        curve = np.mean([r['curve'] for r in runs[c]], axis=0)
        print(f'{c:18s}', ' '.join(f'{v:5.1f}' for v in curve))

    ref, alt = 'spline', 'rational_mv_k3'
    if classes and runs[ref] and runs[alt]:
        print(f'\nper-class final acc (mean over seeds):')
        print(f"{'class':12s} {ref:>9s} {alt:>15s} {'diff':>7s}")
        for i, cls in enumerate(classes):
            a = np.mean([r['final'][i] for r in runs[ref]])
            b = np.mean([r['final'][i] for r in runs[alt]])
            print(f'{cls:12s} {a:9.1f} {b:15.1f} {b - a:+7.1f}')

    print('\nWelch t-tests (final acc):')
    for a, b in TESTS:
        if min(len(runs[a]), len(runs[b])) < 2:
            continue
        x = np.array([r['final'][-1] for r in runs[a]])
        y = np.array([r['final'][-1] for r in runs[b]])
        _, p = stats.ttest_ind(x, y, equal_var=False)
        print(f'  {a} vs {b}: {x.mean() - y.mean():+.2f}, p = {p:.3f}')


if __name__ == '__main__':
    main()
