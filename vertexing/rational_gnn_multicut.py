r"""Secondary-vertex reconstruction as edge classification + multicut with a
pure rational-basis graph neural network. Single file: data, model, training,
clustering and evaluation.

Per jet, the tracks form a complete graph. Every edge carries D = 5
pseudo-coordinates from the geometry of the track pair (jet frame, straight
transverse lines):

    d_eta, d_phi             angular separation (antisymmetric)
    L_ij                     signed displacement along the jet axis of the
                             crossing point of the two tracks in the
                             transverse plane (a 2-track vertex candidate)
    dz_ij                    z mismatch of the two tracks at that crossing
    d_z0                     longitudinal impact-parameter difference

each squashed to [0, 1] by u = 1/2 + 1/2 tanh(asinh(delta / s) / 2). Every
layer is a continuous-kernel convolution with K learnable multivariate
rational (safe-Pade) basis functions of u (rational_cnn.MultivariateRationalBasis,
free K, PCA-of-hats init):

    h_i <- h_i + Theta_root h'_i + mean_{j != i} sum_p B_p(u_ij) Theta_p h'_j + b,
    h'  =  ReLU(LayerNorm(h)),

and the edge classifier is a rational kernel too,

    logit_ij = sum_p B'_p(u_ij) [(U h_i)^T diag(w_p) (V h_j) + beta_p] + b,

symmetrised over (i, j). A per-track origin head (pileup / primary / fromBC /
fromB / fromC / fromS / fromTau / secondary) is trained alongside.

Clustering: multicut of the complete graph with edge costs logit_ij - bias
(log-odds; positive = attractive), solved per jet by greedy additive edge
contraction (GAEC), or connected components of {logit_ij > bias} for
comparison. Clusters whose summed origin probability is mostly heavy flavour
are the reconstructed secondary vertices; they are scored against the truth
b/c hadrons like SALT's MaskformerMetrics (one-to-one matching of predicted
and truth vertices, perfect = recall and purity 100 %, loose = both >= 50 %).

Data: the public Delphes ttbar sample of Van Stroud et al. (EPJC 84 (2024)
1020), zenodo.org/records/10371998, in data/vertexing/.

    python vertexing/rational_gnn_multicut.py train --out vertexing/runs/rgnn
    python vertexing/rational_gnn_multicut.py eval --ckpt vertexing/runs/rgnn/best.pt \
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

from rational_cnn import MultivariateRationalBasis

DATA = Path('data/vertexing')
TRACK_VARS = ['d0', 'z0', 'phi_rel', 'eta_rel', 'dr', 'pt_frac', 'charge',
              'signed_2d_ip', 'signed_3d_ip']
JET_VARS = ['pt', 'eta']
ORIGINS = ['pileup', 'primary', 'fromBC', 'fromB', 'fromC', 'fromS',
           'fromTau', 'secondary']
HF = [2, 3, 4]          # fromBC, fromB, fromC
FROM_B = [2, 3]
CRITERIA = {'perfect': (1.0, 1.0), 'loose': (0.5, 0.5)}
PAIR_NAMES = ['d_eta', 'd_phi', 'L_ij', 'dz_ij', 'd_z0']


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
    jet_eta = np.broadcast_to(j['eta'][:, None], valid.shape)
    geo = np.stack([c['d0'], c['z0'], c['phi_rel'], c['eta_rel'], jet_eta], -1)
    geo = np.nan_to_num(geo.astype(np.float32)) * valid[..., None]
    t = torch.from_numpy
    return {
        'x': t(x), 'geo': t(geo), 'mask': t(valid.astype(bool)),
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


def pair_geometry(geo, scales):
    r"""Pseudo-coordinates ``[B, L, L, 5]`` in ``[0, 1]`` of every track pair
    (see the module docstring); ``geo`` holds the raw d0, z0, phi_rel,
    eta_rel and jet eta of every track."""
    d0, z0, phi, eta, jet_eta = geo.unbind(-1)
    cos, sin = phi.cos(), phi.sin()
    # Track i in the transverse plane of the jet frame (jet axis = x):
    # r0_i + t a_i, with a_i = (cos phi, sin phi) and the point of closest
    # approach to the beam line r0_i = d0 (sin phi, -cos phi) (Delphes sign
    # convention d0 = x sin phi - y cos phi).
    ax, ay = cos, sin
    rx, ry = d0 * sin, -d0 * cos
    # Crossing r0_i + t_i a_i = r0_j + t_j a_j:
    #   t_i = (dr x a_j) / (a_i x a_j),  t_j = (dr x a_i) / (a_i x a_j).
    drx = rx[:, None, :] - rx[:, :, None]  # [B, i, j] = r0_j - r0_i
    dry = ry[:, None, :] - ry[:, :, None]
    den = ax[:, :, None] * ay[:, None, :] - ay[:, :, None] * ax[:, None, :]
    den = torch.where(den.abs() < 1e-4, torch.full_like(den, 1e-4)
                      * torch.where(den < 0, -1.0, 1.0), den)
    ti = ((drx * ay[:, None, :] - dry * ax[:, None, :]) / den).clamp(-1e3, 1e3)
    tj = ((drx * ay[:, :, None] - dry * ax[:, :, None]) / den).clamp(-1e3, 1e3)
    # x of the crossing, averaged over both tracks' sides (identical unless
    # the near-parallel clamps are hit; exactly symmetric)
    L_ij = 0.5 * (rx[:, :, None] + ti * ax[:, :, None]
                  + rx[:, None, :] + tj * ax[:, None, :])
    cot = torch.sinh(eta + jet_eta)                       # dz / ds_T
    z_i = z0[:, :, None] + ti * cot[:, :, None]
    z_j = z0[:, None, :] + tj * cot[:, None, :]
    delta = torch.stack([
        eta[:, None, :] - eta[:, :, None],
        phi[:, None, :] - phi[:, :, None],
        L_ij,
        z_j - z_i,
        z0[:, None, :] - z0[:, :, None],
    ], -1)
    s = torch.as_tensor(scales, dtype=delta.dtype, device=delta.device)
    return 0.5 + 0.5 * torch.tanh(torch.asinh(delta / s) / 2)


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

def eval_basis(basis, u):
    r"""``[B, L, L, K]`` basis values at the pseudo-coordinates ``u`` (fp32,
    outside autocast: the safe-Pade denominators are not fp16/bf16 safe)."""
    B, L, _, D = u.shape
    with torch.autocast(device_type=u.device.type, enabled=False):
        return basis(u.reshape(-1, D).float()).view(B, L, L, -1)


class DenseRationalConv(nn.Module):
    r"""SplineConv-type operator with a learnable multivariate rational basis
    on a dense (padded, complete per jet) graph:
    ``x'_i = Theta_root x_i + sum_j A_ij sum_p B_p(u_ij) Theta_p x_j + b``
    with ``A`` the mean-aggregation matrix over valid ``j != i``."""
    def __init__(self, channels, dim, num_bases, kernel_size, degrees, init):
        super().__init__()
        self.basis = MultivariateRationalBasis(
            dim, kernel_size, degrees=degrees, init=init, num_bases=num_bases)
        K = num_bases
        self.weight = nn.Parameter(torch.empty(K, channels, channels))
        self.root = nn.Parameter(torch.empty(channels, channels))
        self.bias = nn.Parameter(torch.empty(channels))
        bound = 1 / math.sqrt(K * channels)  # as SplineConv / RationalConv
        for p in (self.weight, self.root, self.bias):
            nn.init.uniform_(p, -bound, bound)

    def forward(self, x, u, adj):
        B, L, C = x.shape
        K = self.weight.size(0)
        R = eval_basis(self.basis, u)                           # [B, L, L, K]
        xw = x @ self.weight.permute(1, 0, 2).reshape(C, K * C)  # [B, L, K C]
        A = (R * adj[..., None]).reshape(B, L, L * K)
        out = torch.bmm(A.to(xw.dtype), xw.view(B, L * K, C))
        return out + x @ self.root + self.bias


class RationalEdgeHead(nn.Module):
    r"""``logit_ij = sum_p B_p(u_ij) [(U h_i)^T diag(w_p) (V h_j) + beta_p]
    + b``, symmetrised."""
    def __init__(self, channels, rank, dim, num_bases, kernel_size, degrees,
                 init):
        super().__init__()
        self.basis = MultivariateRationalBasis(
            dim, kernel_size, degrees=degrees, init=init, num_bases=num_bases)
        self.U = nn.Linear(channels, rank, bias=False)
        self.V = nn.Linear(channels, rank, bias=False)
        self.w = nn.Parameter(torch.randn(num_bases, rank) / math.sqrt(rank))
        self.beta = nn.Parameter(torch.zeros(num_bases))
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, h, u):
        R = eval_basis(self.basis, u)                     # [B, L, L, K]
        a, c = self.U(h).float(), self.V(h).float()       # [B, L, r]
        logit = torch.einsum('bir,bijr,bjr->bij', a, R @ self.w, c) \
            + R @ self.beta + self.b
        return 0.5 * (logit + logit.transpose(1, 2))


class RationalVertexGNN(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=4, num_bases=12,
                 kernel_size=3, degrees=(4, 3), init='pca', rank=32, dim=5):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden))
        self.convs = nn.ModuleList([
            DenseRationalConv(hidden, dim, num_bases, kernel_size, degrees,
                              init) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden)
                                    for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.edge = RationalEdgeHead(hidden, rank, dim, num_bases,
                                     kernel_size, degrees, init)
        self.origin = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                    nn.Linear(hidden, len(ORIGINS)))

    def forward(self, x, u, mask):
        adj = pair_mask(mask).float()
        adj = adj / adj.sum(-1, keepdim=True).clamp(min=1)
        h = self.embed(x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + conv(F.relu(norm(h)), u, adj)
        h = self.norm(h) * mask[..., None]
        return self.edge(h, u), self.origin(h)


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
    u = pair_geometry(batch['geo'], args.scales)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == 'cuda'):
        logits, origin = model(batch['x'], u, batch['mask'])
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
    return RationalVertexGNN(
        len(TRACK_VARS) + len(JET_VARS), hidden=args.hidden,
        layers=args.layers, num_bases=args.num_bases,
        kernel_size=args.kernel_size, degrees=tuple(args.degrees),
        init=args.init, rank=args.rank, dim=len(PAIR_NAMES))


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
    t.add_argument('--scales', type=float, nargs=5,
                   default=[0.1, 0.1, 0.5, 0.1, 0.1],
                   help='pseudo-coordinate scales for ' + ', '.join(PAIR_NAMES)
                        + ' (rad, rad, mm, mm, mm)')
    t.add_argument('--hidden', type=int, default=128)
    t.add_argument('--layers', type=int, default=4)
    t.add_argument('--num_bases', type=int, default=12)
    t.add_argument('--kernel_size', type=int, default=3,
                   help='hat grid (kernel_size^5) the PCA init refers to')
    t.add_argument('--degrees', type=int, nargs=2, default=[4, 3])
    t.add_argument('--init', default='pca')
    t.add_argument('--rank', type=int, default=32)
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
