r"""Pre-process N-Cars (Sironi et al., CVPR 2018; Prophesee) for the AEGNN
recognition experiment, following ``aegnn/datasets/ncars.py``: time rescaled
by ``beta`` = 0.5e-5 per microsecond, 10000 events sampled per 100 ms
recording, radius graph (r = 3, at most 32 neighbours) built at training time.

N-Cars is distributed by Prophesee behind a request form
(https://www.prophesee.ai/2018/03/13/dataset-n-cars/) and may not be
redistributed. Download and extract it yourself; ``--src`` is the extracted
directory containing ``n-cars_train/{cars,background}/*_td.dat`` and
``n-cars_test/...`` (AEGNN's converted layout ``training/<seq>/events.txt``
with ``is_car.txt`` is supported too). N-Cars has no validation split; the
last ``--val_fraction`` of a fixed random permutation of the training
sequences is held out for model selection.
"""
import argparse
import glob
import json
import os
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from event_utils import write_split  # noqa: E402

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default=osp.join(ROOT, 'data'))
parser.add_argument('--src', type=str, required=True,
                    help='extracted N-Cars directory')
parser.add_argument('--n_samples', type=int, default=10000)
parser.add_argument('--beta', type=float, default=0.5e-5)
parser.add_argument('--val_fraction', type=float, default=0.1)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=0)
CLASSES = ['background', 'car']


def collect(src, split):
    r"""(files, labels) for the Prophesee layout or AEGNN's converted one."""
    files, labels = [], []
    for name in [f'n-cars_{split}', split, {'train': 'training'}.get(split,
                                                                      split)]:
        d = osp.join(src, name)
        if not osp.isdir(d):
            continue
        for c, cls in enumerate(['background', 'cars']):
            fs = sorted(glob.glob(osp.join(d, cls, '*.dat')))
            files += fs
            labels += [c] * len(fs)
        for seq in sorted(glob.glob(osp.join(d, '*', 'events.txt'))):
            with open(osp.join(osp.dirname(seq), 'is_car.txt')) as f:
                labels.append(int(f.read().strip() == '1'))
            files.append(seq)
        if files:
            return files, labels
    raise FileNotFoundError(f'no {split} sequences under {src}')


if __name__ == '__main__':
    args = parser.parse_args()
    processed_dir = osp.join(args.root, 'NCars', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    with open(osp.join(processed_dir, 'classes.json'), 'w') as f:
        json.dump(CLASSES, f)

    files, labels = collect(args.src, 'train')
    perm = np.random.default_rng(args.seed).permutation(len(files))
    n_val = int(round(args.val_fraction * len(files)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    splits = {
        'training': ([files[i] for i in train_idx], [labels[i] for i in train_idx]),
        'validation': ([files[i] for i in val_idx], [labels[i] for i in val_idx]),
        'test': collect(args.src, 'test'),
    }
    for split, (fs, ls) in splits.items():
        write_split(processed_dir, split, fs, ls, args.n_samples, args.beta,
                    workers=args.workers, seed=args.seed, rel_to=args.src)
    print('done')
