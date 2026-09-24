r"""PascalVOC-Keypoints matching with Deep Graph Matching Consensus (Fey et
al., ICLR 2020), with a SplineCNN or rational-basis backbone. Run
`python experiments/prepare_pascal_voc.py` once first (dataset download +
VGG16 feature extraction)."""
import os.path as osp
import sys
import time

import argparse
import torch
from torch_geometric.datasets import PascalVOCKeypoints as PascalVOC
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
from rational_cnn import DGMC, ValidPairDataset, FaceToEdge  # noqa: E402
from backbones import add_backbone_args, make_backbone, describe, make_optimizer
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
parser.add_argument('--test_samples', type=int, default=1000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save_model', type=str, default='',
                    help='save the trained model (state_dict + args) here')
parser.add_argument('--root', type=str, default=osp.join(ROOT, 'data'),
                    help='dataset root containing PascalVOC/')
add_backbone_args(parser)
add_wandb_args(parser)
args = parser.parse_args()
torch.manual_seed(args.seed)

pre_filter = lambda data: data.pos.size(0) > 0  # noqa
transform = T.Compose([
    T.Delaunay(),
    FaceToEdge(),
    T.Distance() if args.isotropic else T.Cartesian(),
])

train_datasets = []
test_datasets = []
path = osp.join(args.root, 'PascalVOC')
for category in PascalVOC.categories:
    dataset = PascalVOC(path, category, train=True, transform=transform,
                        pre_filter=pre_filter)
    train_datasets += [ValidPairDataset(dataset, dataset, sample=True)]
    dataset = PascalVOC(path, category, train=False, transform=transform,
                        pre_filter=pre_filter)
    test_datasets += [ValidPairDataset(dataset, dataset, sample=True)]
train_dataset = torch.utils.data.ConcatDataset(train_datasets)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                          follow_batch=['x_s', 'x_t'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
psi_1 = make_backbone(args, dataset.num_node_features, args.dim,
                      dataset.num_edge_features, args.num_layers, cat=False,
                      dropout=0.5)
psi_2 = make_backbone(args, args.rnd_dim, args.rnd_dim,
                      dataset.num_edge_features, args.num_layers, cat=True,
                      dropout=0.0)
model = DGMC(psi_1, psi_2, num_steps=args.num_steps).to(device)
print(describe(model))
optimizer = make_optimizer(model, args)
logger = init_wandb(args, 'pascal_voc', model)
meter = StepMeter(model, optimizer)


def generate_y(y_col):
    y_row = torch.arange(y_col.size(0), device=device)
    return torch.stack([y_row, y_col], dim=0)


def train():
    r"""One epoch; returns a dict with the mean total / :math:`S_0` /
    :math:`S_L` losses per graph pair and the :math:`S_L` accuracy."""
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
def test(dataset):
    r"""Returns the :math:`S_0` / :math:`S_L` accuracy and mean per-node
    loss over (at least) :obj:`args.test_samples` keypoints."""
    model.eval()

    loader = DataLoader(dataset, args.batch_size, shuffle=False,
                        follow_batch=['x_s', 'x_t'])

    correct = correct_0 = loss = loss_0 = num_examples = 0
    while (num_examples < args.test_samples):
        for data in loader:
            data = data.to(device)
            S_0, S_L = model(data.x_s, data.edge_index_s, data.edge_attr_s,
                             data.x_s_batch, data.x_t, data.edge_index_t,
                             data.edge_attr_t, data.x_t_batch)
            y = generate_y(data.y)
            correct += model.acc(S_L, y, reduction='sum')
            correct_0 += model.acc(S_0, y, reduction='sum')
            loss += model.loss(S_L, y, reduction='sum').item()
            loss_0 += model.loss(S_0, y, reduction='sum').item()
            num_examples += y.size(1)

            if num_examples >= args.test_samples:
                return {'acc': correct / num_examples,
                        'acc_0': correct_0 / num_examples,
                        'loss': loss / num_examples,
                        'loss_0': loss_0 / num_examples}


best = 0
for epoch in range(1, args.epochs + 1):
    t = time.perf_counter()
    tr = train()
    train_time = time.perf_counter() - t
    print(f'Epoch: {epoch:02d}, Loss: {tr["loss"]:.4f}, Acc: {tr["acc"]:.2f}, '
          f'Time: {train_time:.1f}s')

    t = time.perf_counter()
    results = [test(test_dataset) for test_dataset in test_datasets]
    test_time = time.perf_counter() - t
    accs = [100 * r['acc'] for r in results]
    accs += [sum(accs) / len(accs)]
    best = max(best, accs[-1])
    mean = lambda key: sum(r[key] for r in results) / len(results)  # noqa

    print(' '.join([c[:5].ljust(5) for c in PascalVOC.categories] + ['mean']))
    print(' '.join([f'{acc:.1f}'.ljust(5) for acc in accs]))
    logger.log({**{f'train/{k}': v for k, v in tr.items()},
                'train/time': train_time,
                'test/acc': accs[-1], 'test/acc_0': 100 * mean('acc_0'),
                'test/loss': mean('loss'), 'test/loss_0': mean('loss_0'),
                'test/best_acc': best, 'test/time': test_time,
                **{f'test/acc_{c}': a for c, a in
                   zip(PascalVOC.categories, accs)},
                **meter.metrics()}, step=epoch)
logger.summary({'final_acc': accs[-1], 'best_acc': best})
logger.finish()
if args.save_model:
    torch.save({'state_dict': model.state_dict(), 'args': vars(args),
                'num_node_features': dataset.num_node_features,
                'final_acc': accs[-1]}, args.save_model)
    print(f'Saved model to {args.save_model}')
