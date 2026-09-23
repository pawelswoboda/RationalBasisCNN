r"""SPair-71k keypoint matching with Deep Graph Matching Consensus (Fey et al.,
ICLR 2020) and a SplineCNN or rational-basis backbone: the PascalVOC
experiment (experiments/pascal_voc.py) on the SPair-71k benchmark.

Protocol (pygmtools' SPair71k, two-graph matching): the fixed trn/val/test
pairs of the `large` layout, no difficulty filtering, and only the keypoints
visible in both images. Target keypoints are shuffled -- randomly for
training, with one fixed permutation per pair for evaluation -- because the
annotated order is the same in both images. Accuracy is reported per category
and averaged over the 18 categories, pair-averaged (the benchmark convention)
and keypoint-weighted (the pascal_voc.py convention), on val and test after
every epoch. Run `python experiments/prepare_spair71k.py` once first."""
import os
import os.path as osp
import sys
import time

import argparse
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
# Redirected to project storage by hpc/env.sh; $HOME is quota-limited.
DEFAULT_ROOT = os.environ.get('RBCNN_DATA_ROOT', osp.join(ROOT, 'data'))
from rational_cnn import DGMC, FaceToEdge  # noqa: E402
from rational_cnn.spair import (SPair71k, SPair71kPairs,  # noqa: E402
                                MatchingAccuracy)
from backbones import (add_backbone_args, make_backbone, describe,
                       make_optimizer, freeze_basis)
from wandb_util import add_wandb_args, init_wandb, StepMeter

parser = argparse.ArgumentParser()
parser.add_argument('--isotropic', action='store_true')
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--rnd_dim', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--layout', type=str, default='large',
                    choices=['large', 'small'])
parser.add_argument('--eval_every', type=int, default=1,
                    help='evaluate on val and test every N epochs (and last)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='DataLoader workers (graphs are built on the fly)')
parser.add_argument('--root', type=str, default=DEFAULT_ROOT,
                    help='dataset root containing SPair71k/ (processed graphs)')
parser.add_argument('--spair_root', type=str,
                    default=os.environ.get('RBCNN_SPAIR_ROOT'),
                    help='extracted SPair-71k tree, only read if SPair71k/ is '
                    'not processed yet (default: <root>/SPair-71k)')
add_backbone_args(parser)
add_wandb_args(parser)
args = parser.parse_args()
torch.manual_seed(args.seed)

dataset = SPair71k(osp.join(args.root, 'SPair71k'),
                   raw_dir=args.spair_root or osp.join(args.root, 'SPair-71k'))
transform = T.Compose([
    T.Delaunay(),
    FaceToEdge(),
    T.Distance() if args.isotropic else T.Cartesian(),
])
edge_dim = 1 if args.isotropic else 2

train_dataset = SPair71kPairs(dataset, 'trn', args.layout, transform, 'random')
val_dataset = SPair71kPairs(dataset, 'val', args.layout, transform, 'fixed')
test_dataset = SPair71kPairs(dataset, 'test', args.layout, transform, 'fixed')
print(f'SPair-71k ({args.layout} layout): {len(train_dataset)} train / '
      f'{len(val_dataset)} val / {len(test_dataset)} test pairs')

loader_args = dict(follow_batch=['x_s', 'x_t'], num_workers=args.num_workers,
                   persistent_workers=args.num_workers > 0)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                          **loader_args)
val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False,
                        **loader_args)
test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False,
                         **loader_args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
psi_1 = make_backbone(args, dataset.num_node_features, args.dim, edge_dim,
                      args.num_layers, cat=False, dropout=0.5)
psi_2 = make_backbone(args, args.rnd_dim, args.rnd_dim, edge_dim,
                      args.num_layers, cat=True, dropout=0.0)
model = DGMC(psi_1, psi_2, num_steps=args.num_steps).to(device)
print(describe(model))
frozen = freeze_basis(model, args)
if frozen:
    print(f'froze {frozen} basis-shape parameters (--freeze_basis)')
optimizer = make_optimizer(model, args)
logger = init_wandb(args, 'spair71k', model)
meter = StepMeter(model, optimizer)


def generate_y(y_col):
    y_row = torch.arange(y_col.size(0), device=device)
    return torch.stack([y_row, y_col], dim=0)


def train():
    r"""One epoch over all training pairs; returns the mean total / S_0 / S_L
    losses per graph pair and the keypoint-weighted S_L accuracy."""
    model.train()

    total_loss = total_loss_0 = total_loss_L = correct = num_nodes = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        S_0, S_L = model(data.x_s, data.edge_index_s, data.edge_attr_s,
                         data.x_s_batch, data.x_t, data.edge_index_t,
                         data.edge_attr_t, data.x_t_batch)
        y = generate_y(data.y)
        loss_0 = model.loss(S_0, y)
        loss_L = model.loss(S_L, y) if model.num_steps > 0 else loss_0
        loss = loss_0 + loss_L if model.num_steps > 0 else loss_0
        loss.backward()
        meter.step()
        num_graphs = data.x_s_batch.max().item() + 1
        total_loss += loss.item() * num_graphs
        total_loss_0 += loss_0.item() * num_graphs
        total_loss_L += loss_L.item() * num_graphs
        correct += model.acc(S_L, y, reduction='sum')
        num_nodes += y.size(1)

    n = len(train_loader.dataset)
    return {'loss': total_loss / n, 'loss_0': total_loss_0 / n,
            'loss_L': total_loss_L / n, 'acc': correct / num_nodes}


@torch.no_grad()
def evaluate(loader):
    r"""Per-category S_L accuracy over all pairs of :obj:`loader` (pair-
    averaged and keypoint-weighted), the S_0 category means, and the mean
    per-keypoint losses."""
    model.eval()

    acc_0 = MatchingAccuracy(len(SPair71k.categories))
    acc_L = MatchingAccuracy(len(SPair71k.categories))
    loss = loss_0 = num_nodes = 0
    for data in loader:
        data = data.to(device)
        S_0, S_L = model(data.x_s, data.edge_index_s, data.edge_attr_s,
                         data.x_s_batch, data.x_t, data.edge_index_t,
                         data.edge_attr_t, data.x_t_batch)
        y = generate_y(data.y)
        loss += model.loss(S_L, y, reduction='sum').item()
        loss_0 += model.loss(S_0, y, reduction='sum').item()
        num_nodes += y.size(1)
        acc_L.update(S_L[y[0]].argmax(dim=-1), y[1], data.x_s_batch,
                     data.category)
        acc_0.update(S_0[y[0]].argmax(dim=-1), y[1], data.x_s_batch,
                     data.category)

    out, out_0 = acc_L.compute(), acc_0.compute()
    return {**out, 'pair_mean_0': out_0['pair_mean'],
            'kp_mean_0': out_0['kp_mean'], 'loss': loss / num_nodes,
            'loss_0': loss_0 / num_nodes}


def row(label, values):
    return ' '.join([label.ljust(7)] + [f'{v:.1f}'.ljust(5) for v in values])


header = ' '.join(['split'.ljust(7)] +
                  [c[:5].ljust(5) for c in SPair71k.categories] + ['mean'])
eval_epochs, val_curve, test_curve, kp_curve = [], [], [], []
for epoch in range(1, args.epochs + 1):
    t = time.perf_counter()
    tr = train()
    train_time = time.perf_counter() - t
    line = (f'Epoch: {epoch:02d}, Loss: {tr["loss"]:.4f}, '
            f'Acc: {tr["acc"]:.2f}, Time: {train_time:.1f}s')
    metrics = {**{f'train/{k}': v for k, v in tr.items()},
               'train/time': train_time}

    if epoch % args.eval_every == 0 or epoch == args.epochs:
        t = time.perf_counter()
        val = evaluate(val_loader)
        test = evaluate(test_loader)
        eval_time = time.perf_counter() - t
        print(f'{line}, Eval: {eval_time:.1f}s')
        print(header)
        print(row('val', val['pair'] + [val['pair_mean']]))
        print(row('test', test['pair'] + [test['pair_mean']]))
        print(row('test_kp', test['kp'] + [test['kp_mean']]))
        eval_epochs.append(epoch)
        val_curve.append(val['pair_mean'])
        test_curve.append(test['pair_mean'])
        kp_curve.append(test['kp_mean'])
        best = max(range(len(val_curve)), key=val_curve.__getitem__)
        for name, res in (('val', val), ('test', test)):
            metrics.update({
                f'{name}/acc': res['pair_mean'],
                f'{name}/acc_kp': res['kp_mean'],
                f'{name}/acc_0': res['pair_mean_0'],
                f'{name}/acc_kp_0': res['kp_mean_0'],
                f'{name}/loss': res['loss'], f'{name}/loss_0': res['loss_0'],
            })
        metrics.update({
            'test/best_acc': max(test_curve),
            'test/acc_at_best_val': test_curve[best],
            'test/eval_time': eval_time,
            **{f'test/acc_{c}': a for c, a in
               zip(SPair71k.categories, test['pair'])},
        })
    else:
        print(line)
    logger.log({**metrics, **meter.metrics()}, step=epoch)

best = max(range(len(val_curve)), key=val_curve.__getitem__)
print(f'Final test accuracy: {test_curve[-1]:.2f} (best {max(test_curve):.2f}, '
      f'at best val epoch {eval_epochs[best]}: {test_curve[best]:.2f})')
print(f'Final test keypoint accuracy: {kp_curve[-1]:.2f}')
logger.summary({'final_acc': test_curve[-1], 'best_acc': max(test_curve),
                'acc_at_best_val': test_curve[best],
                'best_val_epoch': eval_epochs[best],
                'final_acc_kp': kp_curve[-1]})
logger.finish()
