r"""Plots learned 2-D rational basis functions of a trained keypoint matching
model on the unit square, next to their initialisation (the B-spline hats
they were fitted to, or the PCA components of the hats).

    # DGMC model saved by experiments/pascal_voc.py --save_model
    python experiments/plot_basis.py results/models/voc_mv_K6_seed0.pt \
        --layers psi_1.convs.0,psi_1.convs.1 --out paper/figures/basis_voc_mv_K6.pdf

    # NMT geometric-refinement GNN (psi_final.pt written by the NMT training
    # script; contains the RationalConv state dict and the SPLINE_CNN options)
    python experiments/plot_basis.py .../psi_final.pt --nmt \
        --layers convs.0,convs.1 --out paper/figures/basis_nmt_spair_mv_K6.pdf

The initialisation is obtained by ``reset_parameters()`` on a copy of the
trained layer (the hat / PCA fit is deterministic).
"""
import argparse
import copy
import os.path as osp
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))


def grid(n):
    t = (torch.arange(n, dtype=torch.float32) + 0.5) / n
    uu, vv = torch.meshgrid(t, t, indexing='xy')
    return torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)  # [n*n, 2]


@torch.no_grad()
def evaluate(conv, n):
    r"""Basis of one :class:`RationalConv` on an n x n grid, shape
    [K, n, n] (row index = second coordinate, column = first)."""
    conv = conv.cpu().eval()
    B = conv.basis(grid(n))  # [n*n, K]
    return B.t().reshape(-1, n, n).numpy()


@torch.no_grad()
def evaluate_init(conv, n):
    fresh = copy.deepcopy(conv).cpu()
    fresh.reset_parameters()
    return evaluate(fresh, n)


def plot(rows, titles, out, cmap='RdBu_r'):
    r"""``rows``: list of [K, n, n] arrays, one row of panels each; a common
    symmetric colour scale."""
    K = rows[0].shape[0]
    fig, axes = plt.subplots(len(rows), K, figsize=(1.3 * K + 0.6, 1.4 * len(rows) + 0.3),
                             squeeze=False)
    vmax = max(np.abs(r).max() for r in rows)
    for i, (r, title) in enumerate(zip(rows, titles)):
        for p in range(K):
            ax = axes[i, p]
            im = ax.imshow(r[p], origin='lower', extent=(0, 1, 0, 1), cmap=cmap,
                           vmin=-vmax, vmax=vmax, interpolation='bilinear')
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f'$B_{{{p + 1}}}$', fontsize=9, pad=3)
        axes[i, 0].set_ylabel(title, fontsize=8)
    fig.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.03,
                        wspace=0.08, hspace=0.12)
    cax = fig.add_axes([0.915, 0.12, 0.012, 0.72])
    fig.colorbar(im, cax=cax).ax.tick_params(labelsize=7)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def dgmc_modules(ckpt):
    r"""Rebuilds the DGMC model of ``pascal_voc.py --save_model``."""
    from backbones import make_backbone
    from rational_cnn.dgmc import DGMC
    a = argparse.Namespace(**ckpt['args'])
    dim = 1 if a.isotropic else 2
    psi_1 = make_backbone(a, ckpt['num_node_features'], a.dim, dim,
                          a.num_layers, cat=False, dropout=0.5)
    psi_2 = make_backbone(a, a.rnd_dim, a.rnd_dim, dim, a.num_layers,
                          cat=True, dropout=0.0)
    model = DGMC(psi_1, psi_2, num_steps=a.num_steps)
    model.load_state_dict(ckpt['state_dict'])
    return dict(model.named_modules())


def nmt_modules(ckpt):
    r"""Rebuilds the two RationalConv layers of the NMT ``SConv`` from
    ``psi_final.pt`` (state dict + SPLINE_CNN options)."""
    from rational_cnn.rational import RationalConv
    sc = ckpt['spline_cnn']
    state = ckpt['state_dict']
    mods, in_ch = {}, sc['input_features']
    for i in range(2):
        conv = RationalConv(in_ch, sc['output_features'], 2, sc.get('kernel_size', 5),
                            aggr=sc.get('aggr', 'max'), pyg_init=True,
                            basis=sc.get('rational_basis', 'multivariate'),
                            degrees=tuple(sc.get('degrees', (8, 6))),
                            init=sc.get('init', 'spline'), vp=sc.get('vp', False),
                            num_bases=sc.get('num_bases', 0) or None,
                            safe=sc.get('safe', 'B'), poly=sc.get('poly', 'chebyshev'))
        prefix = f'convs.{i}.'
        conv.load_state_dict({k[len(prefix):]: v for k, v in state.items()
                              if k.startswith(prefix)})
        mods[f'convs.{i}'] = conv
        in_ch = sc['output_features']
    return mods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--out', required=True)
    parser.add_argument('--n', type=int, default=101, help='grid resolution')
    parser.add_argument('--layers', type=str, default='psi_1.convs.0,psi_1.convs.1',
                        help='comma-separated RationalConv module names')
    parser.add_argument('--nmt', action='store_true',
                        help='checkpoint is psi_final.pt of the NMT script')
    parser.add_argument('--no_init', action='store_true',
                        help='do not plot the initialisation row')
    parser.add_argument('--titles', type=str, default='',
                        help='comma-separated row labels for the layers')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    modules = nmt_modules(ckpt) if args.nmt else dgmc_modules(ckpt)
    names = args.layers.split(',')
    labels = args.titles.split(',') if args.titles else names

    rows, titles = [], []
    for name, label in zip(names, labels):
        conv = modules[name]
        if not args.no_init:
            rows.append(evaluate_init(conv, args.n))
            titles.append(f'{label}\ninit')
        rows.append(evaluate(conv, args.n))
        titles.append(f'{label}\nlearned')
    plot(rows, titles, args.out)
    print(f'wrote {args.out}: layers {names}, K = {rows[0].shape[0]}, '
          f'max |B| = {max(np.abs(r).max() for r in rows):.2f}')


if __name__ == '__main__':
    main()
