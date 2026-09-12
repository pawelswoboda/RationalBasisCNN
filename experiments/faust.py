r"""FAUST shape correspondence (SplineCNN, Fey et al. 2018, Sec. 5.2;
PyG examples/faust.py): 6 continuous-kernel conv layers on 3-D Cartesian
pseudo-coordinates, predicting for every vertex of a test mesh its index on
the reference mesh. Requires `MPI-FAUST.zip` (registration at
http://faust.is.tue.mpg.de/) in `data/FAUST/raw/`; see
`experiments/prepare_faust.py`. `--backbone spline`
reproduces the reference model (kernel 5^3 = 125 basis functions),
`--backbone rational --rational_basis multivariate --num_bases K --init pca`
replaces the basis with K learnable rational functions."""
import os.path as osp
import sys
import time

import argparse
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric.datasets.faust as faust_module
from torch_geometric.datasets import FAUST
from torch_geometric.loader import DataLoader

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
from rational_cnn import FaceToEdge, read_ply  # noqa: E402

faust_module.read_ply = read_ply
from backbones import add_backbone_args, make_conv, describe, make_optimizer
from wandb_util import add_wandb_args, init_wandb, StepMeter

parser = argparse.ArgumentParser()
parser.add_argument('--channels', type=int, nargs='+',
                    default=[32, 64, 64, 64, 64, 64])
parser.add_argument('--aggr', type=str, default='add')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_epoch', type=int, default=61,
                    help='epoch at which the lr is divided by 10')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--root', type=str, default=osp.join(ROOT, 'data'),
                    help='dataset root containing FAUST/')
parser.add_argument('--eval_every', type=int, default=1)
parser.add_argument('--clip', type=float, default=1.0,
                    help='max gradient norm (0: no clipping). The reference '
                    'protocol without clipping diverges for ~2/3 of the '
                    'seeds with the pure-PyTorch B-spline conv.')
add_backbone_args(parser)
add_wandb_args(parser)
args = parser.parse_args()
torch.manual_seed(args.seed)

path = osp.join(args.root, 'FAUST')
pre_transform = T.Compose([FaceToEdge(), T.Constant(value=1)])
train_dataset = FAUST(path, True, T.Cartesian(), pre_transform)
test_dataset = FAUST(path, False, T.Cartesian(), pre_transform)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, args.batch_size)
d = train_dataset[0]
num_nodes = d.num_nodes


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convs = torch.nn.ModuleList()
        in_channels = 1
        for out_channels in args.channels:
            self.convs.append(make_conv(args, in_channels, out_channels, 3,
                                        aggr=args.aggr))
            in_channels = out_channels
        self.lin1 = torch.nn.Linear(in_channels, 256)
        self.lin2 = torch.nn.Linear(256, num_nodes)

    def forward(self, data):
        x, edge_index, pseudo = data.x, data.edge_index, data.edge_attr
        for conv in self.convs:
            x = F.elu(conv(x, edge_index, pseudo))
        x = F.elu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
print(describe(model))
print(model.convs[0])
target = torch.arange(num_nodes, dtype=torch.long, device=device)
optimizer = make_optimizer(model, args)
logger = init_wandb(args, 'faust', model)
meter = StepMeter(model, optimizer)


def train(epoch):
    model.train()
    if epoch == args.lr_decay_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr / 10

    total_loss = correct = 0
    raw_norms = []
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        y = target.repeat(data.num_graphs)
        loss = F.nll_loss(out, y)
        loss.backward()
        if args.clip > 0:
            raw_norms.append(torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.clip).item())
        meter.step()
        total_loss += loss.item() * data.num_graphs
        correct += out.argmax(dim=1).eq(y).sum().item()
    if raw_norms:
        logger.log({'opt/grad_norm_raw': sum(raw_norms) / len(raw_norms),
                    'opt/grad_norm_raw_max': max(raw_norms)}, step=epoch)
    return total_loss / len(train_dataset), correct / (
        len(train_dataset) * num_nodes)


@torch.no_grad()
def test():
    model.eval()
    correct = total_loss = 0
    for data in test_loader:
        data = data.to(device)
        out = model(data)
        y = target.repeat(data.num_graphs)
        total_loss += F.nll_loss(out, y).item() * data.num_graphs
        correct += out.argmax(dim=1).eq(y).sum().item()
    return correct / (len(test_dataset) * num_nodes), total_loss / len(
        test_dataset)


best = 0
for epoch in range(1, args.epochs + 1):
    t = time.perf_counter()
    loss, train_acc = train(epoch)
    train_time = time.perf_counter() - t
    metrics = {'train/loss': loss, 'train/acc': train_acc,
               'train/time': train_time, **meter.metrics()}
    line = (f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
            f'Time: {train_time:.1f}s')
    if epoch % args.eval_every == 0 or epoch == args.epochs:
        t = time.perf_counter()
        test_acc, test_loss = test()
        best = max(best, test_acc)
        line += f', Test: {test_acc:.4f}, Test loss: {test_loss:.4f}'
        metrics.update({'test/acc': 100 * test_acc, 'test/loss': test_loss,
                        'test/best_acc': 100 * best,
                        'test/time': time.perf_counter() - t})
    print(line)
    logger.log(metrics, step=epoch)
print(f'Final test accuracy: {100 * test_acc:.2f} (best {100 * best:.2f})')
logger.summary({'final_acc': 100 * test_acc, 'best_acc': 100 * best})
logger.finish()
