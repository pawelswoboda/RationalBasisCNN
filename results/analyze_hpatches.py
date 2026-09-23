r"""Aggregates the HPatches transfer runs into a table: mean +- std over seeds,
the effect of the refinement against its controls, and Welch t-tests between
bases.

Read the *viewpoint* rows. HPatches illumination sequences have identity
homographies, so the two graphs are identical and the task is degenerate there
(a random module fed constant features scores ~100%).

    python results/analyze_hpatches.py [--results DIR] [--split viewpoint]
"""
import argparse
import glob
import json
import os
import os.path as osp
import re
from collections import defaultdict

import numpy as np
from scipy import stats

ARMS = ['raw', 'trained', 'random', 'mean', 'geometry', 'geometry_random']
ORDER = ['spline', 'spline_k3', 'spline_k2', 'rational', 'rational_k3',
         'rational_mv', 'rational_mv_k3', 'mv_K1', 'mv_K2', 'mv_K4', 'mv_K6',
         'mv_K9', 'mv_K16', 'mlp_K4', 'mlp_K9', 'rational_mv_d54']


def sd(x):
    return x.std(ddof=1) if len(x) > 1 else float('nan')


def main():
    default = osp.join(os.environ.get('RBCNN_PFS_ROOT', '.'), 'hpatches',
                       'k128_d1.0')
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default=default,
                        help='directory of <config>_seed<N>.json files')
    parser.add_argument('--split', default='viewpoint',
                        choices=['viewpoint', 'illumination', 'all'])
    parser.add_argument('--metric', default='acc')
    args = parser.parse_args()

    # values in the json are already percentages
    runs = defaultdict(dict)   # config -> seed -> {arm: percent}
    meta = {}
    for path in sorted(glob.glob(osp.join(args.results, '*_seed*.json'))):
        m = re.fullmatch(r'(.+)_seed(\d+)\.json', osp.basename(path))
        if not m:
            continue
        with open(path) as f:
            blob = json.load(f)
        config, seed = m.group(1), int(m.group(2))
        key = f'{{}}/{args.split}'
        runs[config][seed] = {
            arm: blob['results'][key.format(arm)][args.metric]
            for arm in ARMS if key.format(arm) in blob['results']}
        meta[config] = blob

    print('=' * 92)
    print(f'HPatches out-of-task transfer -- {args.split} pairs, '
          f'metric "{args.metric}" (%)')
    if not runs:
        print(f'no results in {args.results}; run slurm/submit_hpatches.sh')
        return
    any_blob = next(iter(meta.values()))
    chance = any_blob['results'][f'raw/{args.split}'].get('chance', float('nan'))
    print(f"{any_blob['num_keypoints']} keypoints per image, "
          f"{any_blob['distractors']:g}x distractors, chance {chance:.2f}%")
    print('arms: raw = no refinement | trained = the claim | random, mean, '
          'geometry = controls')

    configs = [c for c in ORDER if c in runs] + sorted(set(runs) - set(ORDER))
    present = [a for a in ARMS if any(a in v for s in runs.values()
                                      for v in s.values())]
    print()
    print(f"{'config':18s} {'seeds':>5s} " +
          ' '.join(f'{a:>16s}' for a in present))
    for config in configs:
        seeds = runs[config]
        cells = []
        for arm in present:
            vals = np.array([s[arm] for s in seeds.values() if arm in s])
            cells.append(f'{vals.mean():7.2f} ± {sd(vals):4.2f}'
                         if len(vals) else f'{"-":>16s}')
        print(f'{config:18s} {len(seeds):5d} ' + ' '.join(cells))

    print('\neffect of the refinement (paired over seeds):')
    print(f"{'config':18s} {'trained - raw':>20s} {'trained - random':>20s} "
          f"{'trained - mean':>20s}")
    for config in configs:
        seeds = runs[config]
        row = f'{config:18s}'
        for control in ('raw', 'random', 'mean'):
            a = np.array([s['trained'] for s in seeds.values()
                          if 'trained' in s and control in s])
            b = np.array([s[control] for s in seeds.values()
                          if 'trained' in s and control in s])
            if len(a) < 2:
                row += f'{"-":>21s}'
                continue
            d = a - b
            p = stats.ttest_rel(a, b).pvalue
            row += f'  {d.mean():+7.2f} (p={p:5.3f})'
        print(row)

    base = 'spline'
    if base in runs and len(configs) > 1:
        print(f'\ntransfer of the learnable bases vs "{base}" '
              f'(Welch on the trained arm):')
        x = np.array([s['trained'] for s in runs[base].values()])
        for config in configs:
            if config == base or len(runs[config]) < 2:
                continue
            y = np.array([s['trained'] for s in runs[config].values()])
            p = stats.ttest_ind(y, x, equal_var=False).pvalue
            print(f'  {config:18s} {y.mean() - x.mean():+7.2f}  p = {p:5.3f}')


if __name__ == '__main__':
    main()
