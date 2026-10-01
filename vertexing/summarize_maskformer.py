"""Summarise SALT MaskFormer runs (metrics.csv written by
maskformer_geo.MetricsCSV).

    python vertexing/summarize_maskformer.py vertexing/runs/<dir> [...]
    python vertexing/summarize_maskformer.py --history vertexing/runs/<run>

Each argument is a run directory or a directory of run directories. Per run,
the row of the epoch with the lowest val/loss is reported (as SALT's
checkpoint callback selects it).
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

COLS = {
    'val/loss': 'val loss',
    'val/query_perfect_match_eff': 'perfect eff',
    'val/query_loose_match_eff': 'loose eff',
    'val/query_perfect_match_fake': 'perfect fake',
    'val/query_loose_match_fake': 'loose fake',
    'val/b_eff': 'b eff',
    'val/c_eff': 'c eff',
    'val/jets_classification_loss': 'jet CE',
    'val/query_Lxy_mae': 'Lxy MAE',
}


def runs(paths):
    for p in map(Path, paths):
        if (p / 'metrics.csv').exists():
            yield p
        else:
            yield from sorted(q.parent for q in p.rglob('metrics.csv'))


def variant(run):
    cfg = yaml.safe_load(open(run / 'config.yaml'))
    enc = cfg['model']['model']['init_args']['encoder']
    a = enc.get('init_args', {})
    conv = a.get('conv', enc['class_path'].rsplit('.', 1)[-1])
    if conv == 'spline':
        conv += ' k={}'.format(a['kernel_size'])
    elif conv == 'rational':
        conv += ' K={}'.format(a['num_bases'])
    return conv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--history', action='store_true',
                    help='print every epoch instead of the best one')
    args = ap.parse_args()
    pd.set_option('display.width', 250)
    rows = []
    for run in runs(args.paths):
        df = pd.read_csv(run / 'metrics.csv')
        cols = ['epoch'] + [c for c in COLS if c in df]
        if args.history:
            print(run)
            print(df[cols].rename(columns=COLS).to_string(index=False,
                                                          float_format='%.4f'))
            continue
        best = df.loc[df['val/loss'].idxmin(), cols].to_dict()
        seed = yaml.safe_load(open(run / 'config.yaml')).get('seed_everything')
        best = {'run': run.parent.name, 'variant': variant(run), 'seed': seed,
                'epochs': len(df), **best}
        rows.append(best)
    if rows:
        print(pd.DataFrame(rows).rename(columns=COLS).to_string(
            index=False, float_format='%.4f'))


if __name__ == '__main__':
    main()
