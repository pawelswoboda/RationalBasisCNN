r"""Shared command-line switches for choosing the DGMC backbone (B-spline vs.
rational safe-Padé basis functions) in the example scripts."""
import torch

from rational_cnn import SplineCNN, RationalCNN, BSplineConv, RationalConv


def add_backbone_args(parser):
    parser.add_argument('--backbone', type=str, default='spline',
                        choices=['spline', 'rational'])
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--degrees', type=int, nargs=2, default=[5, 4],
                        help='numerator/denominator degree (rational only)')
    parser.add_argument('--safe', type=str, default='B', choices=['A', 'B'])
    parser.add_argument('--poly', type=str, default='chebyshev',
                        choices=['chebyshev', 'monomial'])
    parser.add_argument('--init', type=str, default='spline',
                        choices=['spline', 'random', 'constant', 'gauss',
                                 'cheb', 'pca'])
    parser.add_argument('--num_bases', type=int, default=0,
                        help='number of basis functions K (multivariate/mlp '
                        'bases; 0 = prod(kernel_size))')
    parser.add_argument('--basis_wd', type=float, default=0.0,
                        help='L2 weight decay on rational basis coefficients')
    parser.add_argument('--vp', action='store_true',
                        help='KAT-style variance-preserving weight init')
    parser.add_argument('--pyg_init', action='store_true',
                        help='PyG SplineConv root/bias init (root bound '
                        '1/sqrt(C_in), zero bias) instead of the kernel bound')
    parser.add_argument('--init_noise', type=float, default=1e-3,
                        help='noise scale of --init constant')
    parser.add_argument('--rational_basis', type=str, default='product',
                        choices=['product', 'multivariate', 'mlp'],
                        help='tensor product of 1-D rational functions, or '
                        'rational functions of multivariate polynomials')
    return parser


def make_backbone(args, in_channels, out_channels, dim, num_layers, cat,
                  dropout):
    if args.backbone == 'spline':
        return SplineCNN(in_channels, out_channels, dim, num_layers, cat=cat,
                         dropout=dropout, kernel_size=args.kernel_size)
    return RationalCNN(in_channels, out_channels, dim, num_layers, cat=cat,
                       dropout=dropout, kernel_size=args.kernel_size,
                       degrees=tuple(args.degrees), safe=args.safe,
                       poly=args.poly, init=args.init,
                       init_noise=args.init_noise, basis=args.rational_basis,
                       vp=args.vp, num_bases=args.num_bases or None)


def make_conv(args, in_channels, out_channels, dim, aggr='mean', **kwargs):
    r"""A single continuous-kernel convolution layer (B-spline or rational
    basis) configured from the command line, for models that stack their
    own layers (e.g. examples/faust.py)."""
    if args.backbone == 'spline':
        return BSplineConv(in_channels, out_channels, dim, args.kernel_size,
                           aggr=aggr, pyg_init=args.pyg_init, **kwargs)
    return RationalConv(in_channels, out_channels, dim, args.kernel_size,
                        aggr=aggr, basis=args.rational_basis, vp=args.vp,
                        pyg_init=args.pyg_init,
                        degrees=tuple(args.degrees), safe=args.safe,
                        poly=args.poly, init=args.init,
                        init_noise=args.init_noise,
                        num_bases=args.num_bases or None, **kwargs)


def make_optimizer(model, args):
    r"""Adam with an optional L2 penalty (:obj:`--basis_wd`) on the rational
    basis coefficients only."""
    wd = getattr(args, 'basis_wd', 0.0)
    if wd <= 0:
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    is_basis = lambda n: n.endswith('numerator') or n.endswith('denominator')  # noqa
    basis = [p for n, p in model.named_parameters() if is_basis(n)]
    rest = [p for n, p in model.named_parameters() if not is_basis(n)]
    return torch.optim.Adam([{'params': rest},
                             {'params': basis, 'weight_decay': wd}],
                            lr=args.lr)


def describe(model):
    num_params = sum(p.numel() for p in model.parameters())
    return f'{model.__class__.__name__} with {num_params} parameters'
