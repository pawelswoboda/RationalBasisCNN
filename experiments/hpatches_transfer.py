r"""Does a keypoint refinement trained for sparse semantic matching transfer,
frozen, to geometric matching it never saw?

Runs the HPatches projection protocol (see rational_cnn/hpatches.py) for four
arms that differ only in how a keypoint's descriptor is produced:

    raw       the frozen VGG16 descriptor, no refinement            (baseline)
    trained   NMT's refinement with its trained weights             (the claim)
    random    the same architecture with untrained weights          (control)
    mean      the mean of the Delaunay neighbours instead of a conv (control)
    geometry  the trained module fed constant features                (control)

`random` separates a learned geometric prior from the shape of the operator,
`mean` separates it from plain smoothing, and `geometry` separates it from a
graph-topology shortcut: with constant inputs, appearance carries no
information at all, so anything above chance (1/N) comes from the Delaunay
structure alone. That control matters because HPatches illumination sequences
have *identity* homographies, which makes the two graphs identical. Comparing a spline run with a
rational run answers whether a *learnable* basis transfers out of task as well
as the fixed B-spline hats.

    python experiments/hpatches_transfer.py --run_dir <NMT run> [--arms ...]
"""
import argparse
import json
import os
import os.path as osp
import sys
import time
from collections import defaultdict

import numpy as np
import torch

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
# Redirected to project storage by hpc/env.sh; $HOME is quota-limited.
DEFAULT_ROOT = os.environ.get('RBCNN_DATA_ROOT', osp.join(ROOT, 'data'))
from rational_cnn.hpatches import (SIZE, KeypointFeatures, MeanAggregation,  # noqa: E402
                                   Refinement, assemble_candidates, build_graph,
                                   distractors,
                                   load_pair, load_state, match, project,
                                   score, sequences, shi_tomasi, to_gray)
import zlib  # noqa: E402
from PIL import Image  # noqa: E402

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
ARMS = ('raw', 'trained', 'random', 'mean', 'geometry', 'geometry_random')

parser = argparse.ArgumentParser()
parser.add_argument('--run_dir', type=str, required=True,
                    help='an NMT run directory holding settings.json and '
                    'params/<epoch>/params.pt')
parser.add_argument('--hpatches_root', type=str,
                    default=osp.join(DEFAULT_ROOT, 'HPatches'))
parser.add_argument('--arms', nargs='+', default=list(ARMS), choices=ARMS)
parser.add_argument('--num_keypoints', type=int, default=128,
                    help='keypoints detected per reference image; NMT trained '
                    'on graphs of ~7-20 nodes, so this is already a stretch')
parser.add_argument('--min_keypoints', type=int, default=16)
parser.add_argument('--distractors', type=float, default=1.0,
                    help='distractor candidates per query, detected '
                    'independently in the target image. 0 reproduces the '
                    'leaky protocol where the candidates are exactly the '
                    'homography image of the queries')
parser.add_argument('--distractor_min_dist', type=float, default=6.0,
                    help='pixels a distractor must keep from every true '
                    'correspondence, so it is not silently a correct answer')
parser.add_argument('--max_sequences', type=int, default=None,
                    help='debugging: only the first N sequences')
parser.add_argument('--device', type=str, default=None)
parser.add_argument('--out', type=str, default=None, help='write JSON here')
args = parser.parse_args()

device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

state_run = args.run_dir.rstrip('/')
trained, settings = Refinement.from_checkpoint(state_run)
random_arm, _ = Refinement.from_checkpoint(state_run, randomize=True)
# The fine-tuned VGG16 lives in the same checkpoint, so every arm sees the
# identical backbone and only the refinement stage varies.
extractor = KeypointFeatures(load_state(state_run)).to(device)
trained, random_arm = trained.to(device), random_arm.to(device)
mean_arm = MeanAggregation().to(device)

cfg = settings['SPLINE_CNN']
print(f'run        : {state_run}')
print(f'refinement : conv={cfg["conv"]} basis={cfg.get("rational_basis")} '
      f'kernel={cfg.get("kernel_size")} K={cfg.get("num_bases") or "k**dim"} '
      f'init={cfg.get("init")} -> {sum(p.numel() for p in trained.parameters())/1e6:.2f} M')
print(f'arms       : {" ".join(args.arms)}')
print(f'keypoints  : {args.num_keypoints} per reference image, device {device}')


def prepare(image):
    r"""Resize to the resolution the refinement was trained at and normalise."""
    resized = image.resize((SIZE, SIZE), resample=Image.Resampling.BICUBIC)
    x = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0)
    x = x.permute(2, 0, 1)
    x = (x - torch.tensor(NORM_MEAN).view(3, 1, 1)) / torch.tensor(NORM_STD).view(3, 1, 1)
    scale = np.array([SIZE / image.size[0], SIZE / image.size[1]])
    return x.unsqueeze(0).to(device), scale


def descriptors(arm, features, edge_index, edge_attr):
    if arm == 'raw':
        return features
    if arm == 'trained':
        return trained(features, edge_index, edge_attr)
    if arm == 'random':
        return random_arm(features, edge_index, edge_attr)
    if arm == 'geometry':
        # constant input: only the graph and its pseudo-coordinates remain
        return trained(torch.ones_like(features), edge_index, edge_attr)
    if arm == 'geometry_random':
        # same, with untrained weights: is even the geometric encoding learned?
        return random_arm(torch.ones_like(features), edge_index, edge_attr)
    return mean_arm(features, edge_index)


results = {arm: defaultdict(list) for arm in args.arms}
seqs = sequences(args.hpatches_root)
if args.max_sequences:
    seqs = seqs[:args.max_sequences]
if not seqs:
    raise SystemExit(f'no HPatches sequences under {args.hpatches_root}')

t0 = time.time()
skipped = 0
for s, (name, kind) in enumerate(seqs):
    ref_image, _, _ = load_pair(args.hpatches_root, name, 2)
    points_ref = shi_tomasi(to_gray(ref_image).to(device), args.num_keypoints)
    if len(points_ref) < args.min_keypoints:
        skipped += 5
        continue
    ref_tensor, ref_scale = prepare(ref_image)
    ref_resized = torch.as_tensor(points_ref.numpy() * ref_scale, dtype=torch.float)
    features_ref_all = extractor(ref_tensor, ref_resized)

    for target in range(2, 7):
        _, trg_image, H = load_pair(args.hpatches_root, name, target)
        projected = project(points_ref.numpy(), H)
        w, h = trg_image.size
        inside = ((projected[:, 0] > 1) & (projected[:, 0] < w - 2) &
                  (projected[:, 1] > 1) & (projected[:, 1] < h - 2))
        if int(inside.sum()) < args.min_keypoints:
            skipped += 1
            continue
        keep = torch.from_numpy(np.flatnonzero(inside))
        gt_points = projected[inside]

        # Candidates = the true correspondences plus keypoints detected
        # independently in the target, so the candidate set is no longer a
        # homography image of the query set.
        extra = distractors(to_gray(trg_image).to(device), gt_points,
                            int(round(args.distractors * len(gt_points))),
                            args.distractor_min_dist)
        seed = zlib.crc32(f'{name}-{target}'.encode()) % (2 ** 31)
        candidates, gt_index = assemble_candidates(gt_points, extra, seed)

        trg_tensor, trg_scale = prepare(trg_image)
        trg_resized = torch.as_tensor(candidates * trg_scale, dtype=torch.float)
        features_trg = extractor(trg_tensor, trg_resized)
        features_ref = features_ref_all[keep]

        edges_ref, attr_ref = build_graph(ref_resized[keep])
        edges_trg, attr_trg = build_graph(trg_resized)
        edges_ref, attr_ref = edges_ref.to(device), attr_ref.to(device)
        edges_trg, attr_trg = edges_trg.to(device), attr_trg.to(device)

        for arm in args.arms:
            a = descriptors(arm, features_ref, edges_ref, attr_ref)
            b = descriptors(arm, features_trg, edges_trg, attr_trg)
            out = score(match(a, b), candidates, gt_index)
            results[arm][kind].append(out)
            results[arm]['all'].append(out)
    if (s + 1) % 20 == 0:
        print(f'  {s + 1}/{len(seqs)} sequences, {time.time() - t0:.0f}s',
              flush=True)

pairs = len(results[args.arms[0]]['all'])
print(f'\n{pairs} image pairs scored ({skipped} skipped for too few keypoints), '
      f'{time.time() - t0:.0f}s')
keys = ['acc', 'mma@1', 'mma@3', 'mma@5']
header = (f"{'arm':15s} {'split':13s} {'pairs':>6s} {'query':>6s} {'cand':>6s} " +
          ' '.join(f'{k:>8s}' for k in keys) + f" {'chance':>8s}")
print(header)
summary = {}
for arm in args.arms:
    for split in ('all', 'viewpoint', 'illumination'):
        rows = results[arm][split]
        if not rows:
            continue
        vals = {k: 100 * float(np.mean([r[k] for r in rows])) for k in keys}
        vals['chance'] = 100 * float(np.mean([1.0 / r['candidates'] for r in rows]))
        summary[f'{arm}/{split}'] = {
            **vals, 'pairs': len(rows),
            'queries': float(np.mean([r['queries'] for r in rows])),
            'candidates': float(np.mean([r['candidates'] for r in rows]))}
        print(f'{arm:15s} {split:13s} {len(rows):6d} '
              f'{np.mean([r["queries"] for r in rows]):6.1f} '
              f'{np.mean([r["candidates"] for r in rows]):6.1f} ' +
              ' '.join(f'{vals[k]:8.2f}' for k in keys) +
              f" {vals['chance']:8.2f}")

if args.out:
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'run_dir': state_run, 'spline_cnn': cfg,
                   'num_keypoints': args.num_keypoints,
                   'distractors': args.distractors,
                   'results': summary}, f, indent=2)
    print(f'\nwrote {args.out}')
