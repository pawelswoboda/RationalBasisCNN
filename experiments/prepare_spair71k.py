r"""Builds the SPair-71k keypoint graphs once -- VGG16 features for all 1,800
images and the trn/val/test pair tables of both layouts -- so that training
jobs only read. Needs the extracted dataset (hpc/download_spair71k.sh) and,
for the VGG16 forward passes, preferably a GPU. Usage:

    python experiments/prepare_spair71k.py [--root data] [--spair_root data/SPair-71k]

writes <root>/SPair71k/processed/{images,pairs}.pt.
"""
import argparse
import os
import os.path as osp
import sys
import time

import torch

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
# Redirected to project storage by hpc/env.sh; $HOME is quota-limited.
DEFAULT_ROOT = os.environ.get('RBCNN_DATA_ROOT', osp.join(ROOT, 'data'))
from rational_cnn.spair import (SPair71k, CATEGORIES, LAYOUTS, SPLITS,  # noqa
                                NUM_PAIRS)

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default=DEFAULT_ROOT,
                    help='writes <root>/SPair71k/processed')
parser.add_argument('--spair_root', type=str,
                    default=os.environ.get('RBCNN_SPAIR_ROOT'),
                    help='extracted SPair-71k tree (default: <root>/SPair-71k)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='images per VGG16 forward pass')
parser.add_argument('--max_images', type=int, default=None,
                    help='debugging: first N images per category only, '
                    'written to processed_maxN/ instead of processed/')
args = parser.parse_args()

raw = args.spair_root or osp.join(args.root, 'SPair-71k')
t = time.time()
dataset = SPair71k(osp.join(args.root, 'SPair71k'), raw_dir=raw,
                   max_images=args.max_images, batch_size=args.batch_size)
pairs = dataset.pairs

num_kps = sum(data.num_nodes for data in dataset)
print(f'{len(dataset)} images, {num_kps} visible keypoints, '
      f'{dataset.num_node_features}-d features -> {dataset.processed_dir}')

ok = True
for layout in LAYOUTS:
    counts = {split: pairs['layout'][layout][split].numel() for split in SPLITS}
    print(f'{layout:5s} layout: ' +
          ' / '.join(f'{n} {split}' for split, n in counts.items()))
    if args.max_images is None and counts != NUM_PAIRS[layout]:
        print(f'  ERROR: expected {NUM_PAIRS[layout]}', file=sys.stderr)
        ok = False

sizes = pairs['ptr'][1:] - pairs['ptr'][:-1]
print(f'keypoints per pair: min {int(sizes.min())}, max {int(sizes.max())}, '
      f'mean {sizes.float().mean():.1f}, single-keypoint pairs '
      f'{int((sizes == 1).sum())}')
test = pairs['layout']['large']['test']
per_cat = torch.bincount(pairs['category'][test], minlength=len(CATEGORIES))
print('large test pairs per category: ' +
      ', '.join(f'{c} {int(n)}' for c, n in zip(CATEGORIES, per_cat)))
print(f'done in {time.time() - t:.0f}s')
sys.exit(0 if ok else 1)
