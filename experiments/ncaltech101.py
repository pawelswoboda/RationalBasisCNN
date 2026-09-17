r"""AEGNN-style object recognition on N-Caltech101 / N-Cars (Schaefer,
Gehrig, Scaramuzza, CVPR 2022) with SplineConv replaced by a rational basis.

Pre-process with ``experiments/prepare_ncaltech101.py`` (or
``prepare_ncars.py`` and ``--dataset ncars``) first. Per batch, the
fixed-size event samples are concatenated on the GPU, the radius graph
(r = 5 / 3, at most 32 neighbours) is built, and pseudo-coordinates are the
Cartesian offsets normalised to [0, 1] (AEGNN ``Cartesian(norm=True,
max_value=r)``). Training follows the paper: Adam, lr 1e-3, batch 16 (64 for
N-Cars), cross-entropy, lr / 10 after 20 epochs.

    python experiments/ncaltech101.py --backbone pyg_spline            # AEGNN
    python experiments/ncaltech101.py --backbone spline                # ours
    python experiments/ncaltech101.py --backbone rational \
        --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 8
    python experiments/ncaltech101.py --backbone pointnet              # Jeziorek et al.
"""
import argparse
import json
import os.path as osp
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from backbones import add_backbone_args, make_optimizer  # noqa: E402
from wandb_util import add_wandb_args, init_wandb  # noqa: E402
from aegnn_net import GraphRes  # noqa: E402

parser = argparse.ArgumentParser()
add_backbone_args(parser)
parser._option_string_actions['--backbone'].choices += ['pyg_spline',
                                                        'pointnet']
parser.set_defaults(kernel_size=2)
parser.add_argument('--channels', type=int, nargs='+',
                    default=[1, 8, 16, 16, 16, 32, 32, 32],
                    help='AEGNN recognition network: 1 8 16 16 16 32 32 32')
parser.add_argument('--dataset', type=str, default='ncaltech101',
                    choices=['ncaltech101', 'ncars'])
parser.add_argument('--aggr', type=str, default='mean',
                    choices=['mean', 'max', 'add'],
                    help='neighbourhood aggregation of spline / rational '
                    'convs (AEGNN: mean); pointnet uses max unless --aggr '
                    'add/max is given explicitly (--pointnet_aggr)')
parser.add_argument('--pointnet_aggr', type=str, default=None,
                    choices=['mean', 'max', 'add'])
parser.add_argument('--radius', type=float, default=None,
                    help='radius graph radius (default 5 / 3)')
parser.add_argument('--max_neighbors', type=int, default=32)
parser.add_argument('--graph', type=str, default='auto',
                    choices=['auto', 'cluster', 'nearest'],
                    help='radius graph via torch_cluster (first 32 '
                    'neighbours found, as AEGNN) or the 32 nearest within r')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay_epoch', type=int, default=20)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=None,
                    help='default 16 / 64')
parser.add_argument('--eval_batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--root', type=str, default=osp.join(ROOT, 'data'))
parser.add_argument('--train_samples', type=int, default=0,
                    help='use only the first n training samples (smoke test)')
parser.add_argument('--eval_every', type=int, default=1)
parser.add_argument('--augment', action='store_true',
                    help='training augmentation: random horizontal flip and '
                    'random translation by up to 10%% of the image size '
                    '(not in AEGNN; the released pipeline has none)')
parser.add_argument('--basis_impl', type=str, default=None,
                    choices=['triton', 'compile', 'eager'],
                    help='rational basis evaluation (default: package '
                    'default, see rational_cnn.triton_basis.DEFAULT_IMPL)')
parser.add_argument('--profile', type=int, default=0,
                    help='time N training batches (graph / forward / '
                    'backward) and exit')
add_wandb_args(parser)
args = parser.parse_args()

DATASETS = {  # AEGNN: image shape, radius, batch size, processed dir
    'ncaltech101': dict(img=(240, 180), r=5.0, bs=16, dir='NCaltech101'),
    'ncars': dict(img=(120, 100), r=3.0, bs=64, dir='NCars'),
}
if args.basis_impl:
    import os
    os.environ['RATIONAL_BASIS_IMPL'] = args.basis_impl
cfg = DATASETS[args.dataset]
if args.backbone == 'pointnet' and args.pointnet_aggr:
    args.aggr = args.pointnet_aggr
args.radius = args.radius or cfg['r']
args.batch_size = args.batch_size or cfg['bs']
IMG_SHAPE = cfg['img']

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------- data
processed = osp.join(args.root, cfg['dir'], 'processed')
with open(osp.join(processed, 'classes.json')) as f:
    classes = json.load(f)


def load_split(split, limit=0):
    pos = np.load(osp.join(processed, f'{split}_pos.npy'), mmap_mode='r')
    x = np.load(osp.join(processed, f'{split}_x.npy'), mmap_mode='r')
    y = np.load(osp.join(processed, f'{split}_y.npy'))
    if limit:
        pos, x, y = pos[:limit], x[:limit], y[:limit]
    return (torch.from_numpy(np.ascontiguousarray(pos)).to(device),
            torch.from_numpy(np.ascontiguousarray(x)).to(device),
            torch.from_numpy(y).to(device))


train = load_split('training', args.train_samples)
val = load_split('validation')
test = load_split('test')
n_events = train[0].size(1)
print(f'{args.dataset}: {len(classes)} classes, {train[2].numel()} train / '
      f'{val[2].numel()} val / {test[2].numel()} test samples, '
      f'{n_events} events each', flush=True)

if args.graph == 'auto':
    try:
        import torch_cluster  # noqa: F401
        args.graph = 'cluster'
    except ImportError:
        args.graph = 'nearest'
print(f'radius graph: {args.graph}', flush=True)
if args.backbone == 'rational':
    import os
    from rational_cnn.triton_basis import DEFAULT_IMPL
    print(f"basis implementation: {os.environ.get('RATIONAL_BASIS_IMPL', DEFAULT_IMPL)}", flush=True)


@torch.no_grad()
def radius_graph_nearest(pos, batch, r, k, num_graphs):
    r"""Directed edges j -> i to the (at most) k nearest nodes j within
    radius r of i, per graph, brute force in chunks on the GPU."""
    n = pos.size(0) // num_graphs
    src, dst = [], []
    for g in range(num_graphs):
        p = pos[g * n:(g + 1) * n]
        for s in range(0, n, 4096):
            q = p[s:s + 4096]
            d2 = torch.cdist(q, p).pow_(2)
            ar = torch.arange(q.size(0), device=pos.device)
            d2[ar, ar + s] = float('inf')  # no self loops
            vals, idx = d2.topk(k, dim=1, largest=False)
            mask = vals <= r * r
            dst_i = (ar + s).unsqueeze(1).expand_as(idx)[mask]
            src.append(idx[mask] + g * n)
            dst.append(dst_i + g * n)
    return torch.stack([torch.cat(src), torch.cat(dst)])


def make_batch(data, idx, augment=False):
    pos_all, x_all, y_all = data
    B = idx.numel()
    pos = pos_all[idx].view(-1, 3)
    x = x_all[idx].view(-1, 1).float()
    batch = torch.arange(B, device=device).repeat_interleave(n_events)
    if augment:
        W, H = IMG_SHAPE
        pos = pos.clone()
        flip = (torch.rand(B, device=device) < 0.5)[batch]
        pos[flip, 0] = (W - 1) - pos[flip, 0]
        shift = (torch.rand(B, 2, device=device) * 2 - 1) * \
            pos.new_tensor([0.1 * W, 0.1 * H])
        pos[:, :2] = pos[:, :2] + shift[batch]
    if args.graph == 'cluster':
        from torch_cluster import radius_graph
        edge_index = radius_graph(pos, r=args.radius, batch=batch, loop=False,
                                  max_num_neighbors=args.max_neighbors)
    else:
        edge_index = radius_graph_nearest(pos, batch, args.radius,
                                          args.max_neighbors, B)
    row, col = edge_index
    edge_attr = (pos[row] - pos[col]) / (2 * args.radius) + 0.5
    return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr,
                batch=batch, y=y_all[idx], num_graphs=B)


# ---------------------------------------------------------------- model
model = GraphRes(args, IMG_SHAPE, len(classes), channels=args.channels).to(
    device)
num_params = sum(p.numel() for p in model.parameters())
conv_params = sum(p.numel() for n, p in model.named_parameters()
                  if n.startswith('conv'))
print(model, flush=True)
print(f'{num_params} parameters ({conv_params} in convolutions)', flush=True)

optimizer = make_optimizer(model, args)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[args.lr_decay_epoch], gamma=0.1)
logger = init_wandb(args, args.dataset, model, conv_params=conv_params)


def train_epoch():
    model.train()
    perm = torch.randperm(train[2].numel(), device=device)
    total_loss, correct, n = 0, 0, 0
    for s in range(0, perm.numel(), args.batch_size):
        idx = perm[s:s + args.batch_size]
        if idx.numel() < 2:
            continue  # BatchNorm
        data = make_batch(train, idx, augment=args.augment)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * idx.numel()
        correct += (out.argmax(-1) == data.y).sum().item()
        n += idx.numel()
    return total_loss / n, 100 * correct / n


@torch.no_grad()
def evaluate(data):
    model.eval()
    correct = 0
    S = data[2].numel()
    for s in range(0, S, args.eval_batch_size):
        idx = torch.arange(s, min(s + args.eval_batch_size, S), device=device)
        batch = make_batch(data, idx)
        correct += (model(batch).argmax(-1) == batch.y).sum().item()
    return 100 * correct / S


if args.profile:
    model.train()
    sync = torch.cuda.synchronize if device.type == 'cuda' else (lambda: None)
    times = np.zeros(3)
    for b in range(args.profile + 1):
        idx = torch.randperm(train[2].numel(), device=device)[:args.batch_size]
        sync()
        t0 = time.time()
        data = make_batch(train, idx)
        sync()
        t1 = time.time()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        sync()
        t2 = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sync()
        t3 = time.time()
        if b > 0:  # warm-up
            times += [t1 - t0, t2 - t1, t3 - t2]
    times /= args.profile
    print(f'{data.edge_index.size(1)} edges/batch; per batch: graph '
          f'{times[0]:.3f}s, forward {times[1]:.3f}s, backward {times[2]:.3f}s'
          f', peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f}GB')
    sys.exit(0)

best_val, test_at_best, best_epoch = 0, 0, 0
for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    loss, train_acc = train_epoch()
    scheduler.step()
    t_train = time.time() - t0
    if epoch % args.eval_every == 0 or epoch == args.epochs:
        val_acc, test_acc = evaluate(val), evaluate(test)
        if val_acc > best_val:
            best_val, test_at_best, best_epoch = val_acc, test_acc, epoch
    else:
        val_acc = test_acc = float('nan')
    mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' \
        else 0
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.2f}, '
          f'Val: {val_acc:.2f}, Test: {test_acc:.2f}, Time: {t_train:.0f}s, '
          f'Mem: {mem:.1f}GB', flush=True)
    logger.log({'loss': loss, 'train_acc': train_acc, 'val_acc': val_acc,
                'test_acc': test_acc, 'epoch_time': t_train,
                'lr': scheduler.get_last_lr()[0]}, step=epoch)

print(f'Best val {best_val:.2f} at epoch {best_epoch}')
print(f'Final test accuracy: {test_acc:.2f} (best {test_at_best:.2f})')
logger.summary({'final_test_acc': test_acc, 'best_val_acc': best_val,
                'test_at_best_val': test_at_best, 'best_epoch': best_epoch,
                'num_params': num_params, 'conv_params': conv_params})
logger.finish()
