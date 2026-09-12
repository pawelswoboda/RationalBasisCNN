r"""Aggregates the training logs in results/logs/ into the tables of the
paper draft:  python results/analyze.py

PascalVOC: per-seed final/best mean-over-categories accuracy (last epoch),
Welch t-tests between the configurations discussed in the paper.
FAUST: per-seed final/best vertex accuracy from the 'Final test accuracy'
line."""
import glob
import os.path as osp
import re
from collections import defaultdict

import numpy as np
from scipy import stats

LOGS = osp.join(osp.dirname(osp.abspath(__file__)), 'logs')


def sd(x):
    return x.std(ddof=1) if len(x) > 1 else float('nan')


def floats(line):
    toks = line.split()
    try:
        return [float(t) for t in toks] if len(toks) >= 3 else None
    except ValueError:
        return None


def welch(res, pairs, value):
    for a, b in pairs:
        if min(len(res[a]), len(res[b])) < 2:
            continue
        x = np.array([value(r) for r in res[a]])
        y = np.array([value(r) for r in res[b]])
        _, p = stats.ttest_ind(x, y, equal_var=False)
        print(f'  {a} vs {b}: {x.mean() - y.mean():+.2f}, p = {p:.3f}')


# ---------------------------------------------------------------- PascalVOC
print('=' * 78)
print('PascalVOC-Keypoints (DGMC, 15 epochs, mean over 20 categories, '
      'test_samples=1000)')
voc = defaultdict(list)
partial = defaultdict(list)
cats = None
for f in sorted(glob.glob(osp.join(LOGS, 'pascal_voc', '*_seed*.log'))):
    cfg, seed = re.match(r'.*/(.*)_seed(\d+)\.log', f).groups()
    lines = open(f).read().splitlines()
    params = next((int(re.search(r'(\d+) parameters', l).group(1))
                   for l in lines if 'parameters' in l), None)
    times = [float(m.group(1)) for l in lines
             for m in [re.search(r'Time: ([\d.]+)s', l)] if m]
    rows = [r for r in (floats(l) for l in lines) if r and len(r) == 21]
    header = next((l for l in lines if l.startswith('aerop')), None)
    if header:
        cats = header.split()[:-1]
    accs = np.array(rows)
    if not any(l.startswith('DONE') for l in lines):
        partial[cfg].append(f'seed{seed}@{len(rows)}ep')
        continue
    voc[cfg].append(dict(params=params, final=accs[-1], best=accs[:, -1].max(),
                         curve=accs[:, -1], time=np.mean(times[1:])))
ORDER = ['spline', 'spline_k3', 'spline_k2', 'rational', 'rational_k3',
         'rational_mv', 'rational_mv_k3', 'mv_K1', 'mv_K2', 'mv_K4', 'mv_K6',
         'mv_K9', 'mv_K16', 'mv_K4_wd', 'mlp_K4', 'mlp_K9', 'rational_mv_d54',
         'rational_mv_k3_gauss', 'rational_mv_k3_cheb_vp']
VOC_CFGS = [c for c in ORDER if voc[c]] + sorted(set(voc) - set(ORDER))
if partial:
    print('incomplete runs (excluded):', dict(partial))
print(f"{'config':24s} {'params':>8s} {'seeds':>5s} {'final acc':>14s} "
      f"{'best acc':>14s} {'s/epoch':>8s}")
for cfg in VOC_CFGS:
    rs = voc[cfg]
    fin = np.array([r['final'][-1] for r in rs])
    best = np.array([r['best'] for r in rs])
    print(f"{cfg:24s} {rs[0]['params']:8d} {len(rs):5d} "
          f"{fin.mean():7.2f} ± {sd(fin):4.2f} {best.mean():7.2f} ± "
          f"{sd(best):4.2f} {np.mean([r['time'] for r in rs]):8.1f}")
print('\nper-seed final acc:')
for cfg in VOC_CFGS:
    print(f'{cfg:24s}', ' '.join(f"{r['final'][-1]:5.1f}" for r in voc[cfg]))
print('\ntest acc per epoch (mean over seeds):')
for cfg in VOC_CFGS:
    c = np.mean([r['curve'] for r in voc[cfg]], axis=0)
    print(f'{cfg:24s}', ' '.join(f'{v:4.1f}' for v in c))
if cats and voc['spline'] and voc['rational_mv_k3']:
    print('\nper-category final acc (mean over seeds):')
    print(f"{'category':11s} {'spline':>7s} {'mv_k3':>7s} {'diff':>6s}")
    for i, c in enumerate(cats):
        s = np.mean([r['final'][i] for r in voc['spline']])
        m = np.mean([r['final'][i] for r in voc['rational_mv_k3']])
        print(f'{c:11s} {s:7.1f} {m:7.1f} {m - s:+6.1f}')
print('\nWelch t-tests (final acc):')
welch(voc, [('rational_mv_k3', 'spline'), ('rational_mv_k3', 'rational_k3'),
            ('rational_mv_k3', 'spline_k3'), ('mv_K4', 'spline'),
            ('mv_K4', 'mlp_K4'), ('mv_K4', 'spline_k2'), ('mv_K9', 'mlp_K9'),
            ('mv_K9', 'rational_mv_k3'), ('mv_K16', 'mv_K9'),
            ('mv_K4_wd', 'mv_K4'), ('rational_mv_d54', 'rational_mv'),
            ('rational_mv_k3_gauss', 'rational_mv_k3'),
            ('rational_mv_k3_cheb_vp', 'rational_mv_k3'),
            ('rational', 'spline'), ('rational_k3', 'spline_k3')],
      lambda r: r['final'][-1])

# -------------------------------------------------------------------- FAUST
print('\n' + '=' * 78)
print('FAUST shape correspondence (exact vertex accuracy, 100 epochs)')
fa = defaultdict(list)
for f in sorted(glob.glob(osp.join(LOGS, 'faust', '*_seed*.log'))):
    cfg = re.match(r'.*/(.*)_seed(\d+)\.log', f).group(1)
    txt = open(f).read()
    m = re.search(r'Final test accuracy: ([\d.]+) \(best ([\d.]+)\)', txt)
    if not m:
        continue
    params = re.search(r'(\d+) parameters', txt)
    fa[cfg].append(dict(final=float(m.group(1)), best=float(m.group(2)),
                        params=int(params.group(1)) if params else 0))
print(f"{'config':26s} {'params':>8s} {'seeds':>5s} {'final acc':>14s} "
      f"{'best acc':>14s}  per-seed")
for cfg, rows in sorted(fa.items(),
                        key=lambda kv: -np.mean([r['final'] for r in kv[1]])):
    fin = np.array([r['final'] for r in rows])
    best = np.array([r['best'] for r in rows])
    print(f"{cfg:26s} {rows[0]['params']:8d} {len(rows):5d} "
          f"{fin.mean():7.2f} ± {sd(fin):4.2f} {best.mean():7.2f} ± "
          f"{sd(best):4.2f}  " + ' '.join(f'{v:5.2f}' for v in fin))
print('\nWelch t-tests (final acc):')
welch(fa, [('faust_mv_K8', 'faust_spline_mean'), ('faust_mv_K4', 'faust_spline_mean'),
           ('faust_mv_K8', 'faust_mlp_K8'), ('faust_mv_K8', 'faust_spline'),
           ('faust_spline_mean', 'faust_spline'),
           ('faust_spline_pyginit', 'faust_spline'),
           ('faust_mv_K4', 'faust_mv_K4_mean')], lambda r: r['final'])
