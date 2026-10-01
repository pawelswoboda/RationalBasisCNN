r"""Secondary-vertex reconstruction as edge classification + multicut with a
Transformer encoder: the "EdgeClassifier" baseline of Van Stroud et al.
(EPJC 84 (2024) 1020; GN1/GN2-style track-pair classification, as SALT's
VertexingTask). Single file: data, model, training, clustering and evaluation.

Per jet, the track features (no pair geometry) are embedded and encoded by a
4-layer Transformer (d = 256, 8 heads, feed-forward 2d, ReLU, pre-norm). The
edge classifier is an MLP on the concatenated embeddings of the two tracks
and the attention-pooled jet representation (hidden layers 128, 64, 32; the
first layer is applied per track and broadcast over pairs), symmetrised over
(i, j). A per-track origin head (pileup / primary / fromBC / fromB / fromC /
fromS / fromTau / secondary) is trained alongside.

Clustering and evaluation are identical to rational_gnn_multicut.py: multicut
of the complete graph with costs logit_ij - bias, solved per jet by greedy
additive edge contraction (GAEC), or connected components of
{logit_ij > bias}; heavy-flavour clusters are scored against the truth b/c
hadrons like SALT's MaskformerMetrics.

Data: the public Delphes ttbar sample, zenodo.org/records/10371998, in
data/vertexing/.

    python vertexing/transformer_edge_multicut.py train --out vertexing/runs/tedge
    python vertexing/transformer_edge_multicut.py eval --ckpt vertexing/runs/tedge/best.pt \
        --file data/vertexing/pp_output_test_ttbar.h5
"""
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from torch import nn


DATA = Path('data/vertexing')
TRACK_VARS = ['d0', 'z0', 'phi_rel', 'eta_rel', 'dr', 'pt_frac', 'charge',
              'signed_2d_ip', 'signed_3d_ip']
JET_VARS = ['pt', 'eta']
ORIGINS = ['pileup', 'primary', 'fromBC', 'fromB', 'fromC', 'fromS',
           'fromTau', 'secondary']
HF = [2, 3, 4]          # fromBC, fromB, fromC
FROM_B = [2, 3]
CRITERIA = {'perfect': (1.0, 1.0), 'loose': (0.5, 0.5)}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

class JetBatches(torch.utils.data.Dataset):
    r"""Batches of consecutive jets read directly from the h5 file (shuffled
    at the batch level by the DataLoader)."""
    def __init__(self, path, num_jets, batch_size, norm_dict):
        self.path = str(path)
        with h5py.File(self.path, 'r') as f:
            n = len(f['jets'])
        self.n = min(num_jets, n) if num_jets else n
        self.batch_size = batch_size
        self.starts = np.arange(0, self.n, batch_size)
        self.norm = yaml.safe_load(open(norm_dict))
        self.file = None

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        if self.file is None:
            self.file = h5py.File(self.path, 'r')
        s = int(self.starts[i])
        e = min(s + self.batch_size, self.n)
        return to_tensors(self.file['consts'][s:e], self.file['jets'][s:e],
                          self.file['hadrons'][s:e], self.norm)


def to_tensors(c, j, h, norm):
    valid = c['valid']
    L = max(int(valid.sum(1).max()), 2)
    c, valid = c[:, :L], valid[:, :L]
    feats = [(c[v] - norm['consts'][v]['mean']) / norm['consts'][v]['std']
             for v in TRACK_VARS]
    feats += [np.broadcast_to(((j[v] - norm['jets'][v]['mean'])
                               / norm['jets'][v]['std'])[:, None], valid.shape)
              for v in JET_VARS]
    x = np.nan_to_num(np.stack(feats, -1).astype(np.float32)) * valid[..., None]
    t = torch.from_numpy
    return {
        'x': t(x), 'mask': t(valid.astype(bool)),
        'vertex': t(np.where(valid, c['truth_vertex_idx'], -1).astype(np.int64)),
        'hadron': t(np.where(valid, c['truth_hadron_idx'], -1).astype(np.int64)),
        'origin': t(np.where(valid, c['truth_origin_label'], -1).astype(np.int64)),
        'hadron_idx': t(np.where(h['valid'], h['hadron_idx'], -1).astype(np.int64)),
        'hadron_flavour': t(np.where(h['valid'], h['flavour'], -1).astype(np.int64)),
    }


def loader(path, num_jets, batch_size, workers, shuffle, norm_dict):
    ds = JetBatches(path, num_jets, batch_size, norm_dict)
    return torch.utils.data.DataLoader(
        ds, batch_size=None, shuffle=shuffle, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0)


def edge_targets(batch, label):
    ids = batch[label]
    return (ids[:, :, None] == ids[:, None, :]) & (ids[:, :, None] >= 0)


def pair_mask(mask, upper=False):
    L = mask.size(1)
    eye = torch.eye(L, dtype=torch.bool, device=mask.device)
    m = mask[:, :, None] & mask[:, None, :] & ~eye
    return m & torch.ones_like(eye).triu(1) if upper else m


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class PairMLPEdgeHead(nn.Module):
    r"""``logit_ij = MLP([h_i, h_j, g])`` with hidden layers ``hidden``; the
    first (linear) layer is split into per-track terms, ``W [h_i; h_j; g] =
    W_1 h_i + W_2 h_j + W_3 g``, so that no ``[B, L, L, 2C]`` tensor is
    built. Symmetrised over (i, j)."""
    def __init__(self, channels, hidden=(128, 64, 32)):
        super().__init__()
        self.first_i = nn.Linear(channels, hidden[0])
        self.first_j = nn.Linear(channels, hidden[0], bias=False)
        self.first_g = nn.Linear(channels, hidden[0], bias=False)
        layers = []
        for a, b in zip(hidden[:-1], hidden[1:]):
            layers += [nn.ReLU(), nn.Linear(a, b)]
        self.rest = nn.Sequential(*layers, nn.ReLU(), nn.Linear(hidden[-1], 1))

    def forward(self, h, g):
        a = self.first_i(h) + self.first_g(g)[:, None]
        z = a[:, :, None] + self.first_j(h)[:, None, :]  # [B, L, L, H]
        logit = self.rest(z).squeeze(-1)
        return 0.5 * (logit + logit.transpose(1, 2))


class TransformerVertexer(nn.Module):
    def __init__(self, in_dim, hidden=256, layers=4, heads=8):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden))
        layer = nn.TransformerEncoderLayer(
            hidden, heads, dim_feedforward=2 * hidden, dropout=0.0,
            activation='relu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.pool = nn.Linear(hidden, 1)  # global attention pooling
        self.edge = PairMLPEdgeHead(hidden)
        self.origin = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                    nn.Linear(hidden, len(ORIGINS)))

    def forward(self, x, mask):
        h = self.encoder(self.embed(x), src_key_padding_mask=~mask)
        h = self.norm(h) * mask[..., None]
        w = self.pool(h).squeeze(-1).masked_fill(~mask, -1e4).softmax(-1)
        g = (w[..., None] * h).sum(1)
        return self.edge(h, g), self.origin(h)


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------

def gaec(cost):
    r"""Greedy additive edge contraction for the multicut problem on a
    complete graph, ``cost`` ``[n, n]`` symmetric (positive = attractive):
    repeatedly contracts the edge of largest positive cost. Returns a cluster
    label per node."""
    n = len(cost)
    C = cost.astype(np.float64).copy()
    np.fill_diagonal(C, -np.inf)
    label = np.arange(n)
    for _ in range(n - 1):
        k = int(np.argmax(C))
        i, j = divmod(k, n)
        if not C[i, j] > 0:
            break
        C[i] += C[j]
        C[:, i] = C[i]
        C[i, i] = -np.inf
        C[j] = -np.inf
        C[:, j] = -np.inf
        label[label == j] = i
    return np.unique(label, return_inverse=True)[1]


def components(cost):
    r"""Connected components of the edges with positive cost."""
    return connected_components(cost > 0, directed=False)[1]


def cluster_metrics(logits, origin_prob, batch, solver, bias, acc):
    r"""Clusters every jet of the batch and accumulates SALT-style vertex
    metrics into ``acc``."""
    logits = logits.float().cpu().numpy()
    origin_prob = origin_prob.float().cpu().numpy()
    n_tracks = batch['mask'].sum(1).numpy()
    hadron = batch['hadron'].numpy()
    hadron_idx = batch['hadron_idx'].numpy()
    flav = batch['hadron_flavour'].numpy()
    solve = gaec if solver == 'multicut' else components
    for b in range(len(n_tracks)):
        n = int(n_tracks[b])
        if n == 0:
            continue
        lab = solve(logits[b, :n, :n] - bias) if n > 1 else np.zeros(1, int)
        p = origin_prob[b, :n]
        # predicted vertices: clusters that are mostly heavy flavour
        onehot = np.eye(lab.max() + 1, dtype=bool)[lab].T       # [m, n]
        score = onehot @ p                                      # [m, 8]
        hf = score[:, HF].sum(1) > score.sum(1) - score[:, HF].sum(1)
        pred = onehot[hf]
        # truth vertices: b / c hadrons with at least one track
        ids = hadron_idx[b][hadron_idx[b] >= 0]
        tgt = hadron[b, :n][None, :] == ids[:, None]            # [t, n]
        keep = tgt.any(1)
        tgt, tflav = tgt[keep], flav[b][hadron_idx[b] >= 0][keep]
        acc['n_pred'] += len(pred)
        acc['n_tgt'] += len(tgt)
        acc['n_tgt_b'] += int((tflav == 5).sum())
        acc['n_tgt_c'] += int((tflav == 4).sum())
        acc['n_clusters'] += int(lab.max() + 1)
        if len(pred) == 0 or len(tgt) == 0:
            continue
        overlap = tgt.astype(np.int64) @ pred.T.astype(np.int64)  # [t, m]
        rows, cols = linear_sum_assignment(-overlap)
        ov = overlap[rows, cols]
        recall = ov / tgt[rows].sum(1)
        purity = ov / pred[cols].sum(1)
        for name, (r, q) in CRITERIA.items():
            ok = (recall >= r) & (purity >= q)
            acc[name] += int(ok.sum())
            acc[name + '_b'] += int((ok & (tflav[rows] == 5)).sum())
            acc[name + '_c'] += int((ok & (tflav[rows] == 4)).sum())


def summarise(acc):
    out = {}
    for name in CRITERIA:
        out[name + '_eff'] = acc[name] / max(acc['n_tgt'], 1)
        out[name + '_fake'] = 1 - acc[name] / max(acc['n_pred'], 1)
        out[name + '_eff_b'] = acc[name + '_b'] / max(acc['n_tgt_b'], 1)
        out[name + '_eff_c'] = acc[name + '_c'] / max(acc['n_tgt_c'], 1)
    out['pred_per_jet'] = acc['n_pred'] / max(acc['n_jets'], 1)
    out['tgt_per_jet'] = acc['n_tgt'] / max(acc['n_jets'], 1)
    return out


# --------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------

def origin_weights(class_dict, cap):
    w = yaml.safe_load(open(class_dict))['consts']['truth_origin_label']
    return torch.tensor(w, dtype=torch.float).clamp(max=cap)


def forward_batch(model, batch, args, device):
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == 'cuda'):
        logits, origin = model(batch['x'], batch['mask'])
    return batch, logits.float(), origin.float()


def losses(batch, logits, origin, args, weights):
    pm = pair_mask(batch['mask'], upper=True)
    tgt = edge_targets(batch, args.edge_label)
    edge = F.binary_cross_entropy_with_logits(logits[pm], tgt[pm].float())
    orig = F.cross_entropy(origin[batch['mask']], batch['origin'][batch['mask']],
                           weight=weights, ignore_index=-1)
    with torch.no_grad():
        pred, t = logits[pm] > 0, tgt[pm]
        tp = (pred & t).sum().item()
        stats = {'tp': tp, 'p': pred.sum().item(), 't': t.sum().item()}
    return edge, orig, stats


@torch.no_grad()
def evaluate(model, data, args, device, weights, solvers, biases,
             max_cluster_jets=None):
    model.eval()
    tot = {'edge': 0.0, 'origin': 0.0, 'n': 0, 'tp': 0, 'p': 0, 't': 0}
    accs = {(s, b): Counter() for s in solvers for b in biases}
    jets = 0
    for batch in data:
        batch_dev, logits, origin = forward_batch(model, batch, args, device)
        edge, orig, st = losses(batch_dev, logits, origin, args, weights)
        tot['edge'] += edge.item()
        tot['origin'] += orig.item()
        tot['n'] += 1
        for k in ('tp', 'p', 't'):
            tot[k] += st[k]
        if max_cluster_jets is None or jets < max_cluster_jets:
            prob = origin.softmax(-1)
            for (s, b), acc in accs.items():
                acc['n_jets'] += len(logits)
                cluster_metrics(logits, prob, batch, s, b, acc)
            jets += len(logits)
    model.train()
    out = {'val_edge_loss': tot['edge'] / tot['n'],
           'val_origin_loss': tot['origin'] / tot['n'],
           'edge_precision': tot['tp'] / max(tot['p'], 1),
           'edge_recall': tot['tp'] / max(tot['t'], 1)}
    out['val_loss'] = out['val_edge_loss'] + args.origin_weight \
        * out['val_origin_loss']
    for (s, b), acc in accs.items():
        out.update({'{}@{:g}/{}'.format(s, b, k): v
                    for k, v in summarise(acc).items()})
    return out


def build_model(args):
    return TransformerVertexer(len(TRACK_VARS) + len(JET_VARS),
                               hidden=args.hidden, layers=args.layers,
                               heads=args.heads)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / 'args.json', 'w'), indent=1, default=str)

    train_data = loader(args.train_file, args.num_train, args.batch_size,
                        args.workers, True, args.norm_dict)
    val_data = loader(args.val_file, args.num_val, args.batch_size,
                      args.workers, False, args.norm_dict)
    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print('parameters: {:,}'.format(n_params))
    weights = origin_weights(args.class_dict, args.origin_weight_cap).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps = args.epochs * len(train_data)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=args.warmup,
        div_factor=args.lr / 1e-7, final_div_factor=1e-7 / 1e-5)  # 1e-7 -> lr -> 1e-5

    best = math.inf
    log = open(out / 'log.jsonl', 'a')
    for epoch in range(args.epochs):
        t0, run = time.time(), {'edge': 0.0, 'origin': 0.0, 'n': 0}
        for it, batch in enumerate(train_data):
            batch, logits, origin = forward_batch(model, batch, args, device)
            edge, orig, _ = losses(batch, logits, origin, args, weights)
            loss = edge + args.origin_weight * orig
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            run['edge'] += edge.item()
            run['origin'] += orig.item()
            run['n'] += 1
            if (it + 1) % args.print_every == 0:
                print('epoch {} it {}/{} edge {:.4f} origin {:.4f} '
                      '{:.0f} jets/s'.format(
                          epoch, it + 1, len(train_data),
                          run['edge'] / run['n'], run['origin'] / run['n'],
                          run['n'] * args.batch_size / (time.time() - t0)),
                      flush=True)
        res = evaluate(model, val_data, args, device, weights,
                       ['multicut', 'cc'], [0.0],
                       max_cluster_jets=args.cluster_val_jets)
        res.update({'epoch': epoch, 'train_edge_loss': run['edge'] / run['n'],
                    'train_origin_loss': run['origin'] / run['n'],
                    'time': time.time() - t0, 'params': n_params})
        log.write(json.dumps(res) + '\n')
        log.flush()
        print(json.dumps(res, indent=1), flush=True)
        state = {'model': model.state_dict(), 'args': vars(args),
                 'epoch': epoch, 'val': res}
        torch.save(state, out / 'last.pt')
        if res['val_loss'] < best:
            best = res['val_loss']
            torch.save(state, out / 'best.pt')


def evaluate_ckpt(args):
    device = torch.device(args.device)
    state = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    targs = argparse.Namespace(**state['args'])
    model = build_model(targs).to(device)
    model.load_state_dict(state['model'])
    weights = origin_weights(targs.class_dict,
                             targs.origin_weight_cap).to(device)
    data = loader(args.file, args.num_jets, targs.batch_size, args.workers,
                  False, targs.norm_dict)
    res = evaluate(model, data, targs, device, weights, args.solvers,
                   args.biases)
    res.update({'ckpt': str(args.ckpt), 'file': str(args.file),
                'num_jets': args.num_jets, 'epoch': state['epoch']})
    print(json.dumps(res, indent=1))
    if args.save:
        json.dump(res, open(args.save, 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)

    t = sub.add_parser('train')
    t.add_argument('--out', required=True)
    t.add_argument('--train_file', default=DATA / 'pp_output_train.h5')
    t.add_argument('--val_file', default=DATA / 'pp_output_val.h5')
    t.add_argument('--norm_dict', default=DATA / 'norm_dict.yaml')
    t.add_argument('--class_dict', default=DATA / 'class_dict.yaml')
    t.add_argument('--num_train', type=int, default=2_000_000)
    t.add_argument('--num_val', type=int, default=200_000)
    t.add_argument('--cluster_val_jets', type=int, default=50_000,
                   help='jets clustered for the per-epoch vertex metrics')
    t.add_argument('--edge_label', choices=['vertex', 'hadron'],
                   default='vertex',
                   help='edge truth: same truth_vertex_idx or same '
                        'truth_hadron_idx (pileup / unlinked: no positives)')
    t.add_argument('--hidden', type=int, default=256)
    t.add_argument('--layers', type=int, default=4)
    t.add_argument('--heads', type=int, default=8)
    t.add_argument('--origin_weight', type=float, default=0.5)
    t.add_argument('--origin_weight_cap', type=float, default=50.0)
    t.add_argument('--epochs', type=int, default=15)
    t.add_argument('--batch_size', type=int, default=1000)
    t.add_argument('--lr', type=float, default=5e-4)
    t.add_argument('--warmup', type=float, default=0.05)
    t.add_argument('--weight_decay', type=float, default=1e-5)
    t.add_argument('--clip', type=float, default=1.0)
    t.add_argument('--workers', type=int, default=6)
    t.add_argument('--print_every', type=int, default=200)
    t.add_argument('--seed', type=int, default=1)
    t.add_argument('--device', default='cuda')

    e = sub.add_parser('eval')
    e.add_argument('--ckpt', required=True)
    e.add_argument('--file', default=DATA / 'pp_output_test_ttbar.h5')
    e.add_argument('--num_jets', type=int, default=500_000)
    e.add_argument('--solvers', nargs='+', default=['multicut', 'cc'],
                   choices=['multicut', 'cc'])
    e.add_argument('--biases', type=float, nargs='+', default=[0.0],
                   help='edge-cost offsets: cost = logit - bias')
    e.add_argument('--workers', type=int, default=6)
    e.add_argument('--save')
    e.add_argument('--device', default='cuda')

    args = ap.parse_args()
    train(args) if args.cmd == 'train' else evaluate_ckpt(args)


if __name__ == '__main__':
    main()
