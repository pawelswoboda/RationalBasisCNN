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
                        choices=['product', 'multivariate', 'mlp', 'regional'],
                        help='tensor product of 1-D rational functions, '
                        'rational functions of multivariate polynomials, or '
                        'multivariate rationals confined to radial zones')
    parser.add_argument('--zones', type=int, default=4,
                        help='radial zones of the regional basis; '
                        'K = zones * bases_per_zone')
    parser.add_argument('--bases_per_zone', type=int, default=1,
                        help='rational functions inside each zone')
    parser.add_argument('--zone_overlap', type=float, default=2.0,
                        help='how far a zone window reaches past its own ring, '
                        'in half-widths (must exceed 1)')
    parser.add_argument('--freeze_basis', action='store_true',
                        help='hold the basis shapes at their initialisation '
                        'and train only the kernel weights; supplies the '
                        'fixed-basis cell of the fixed/learned x local/global '
                        'design')
    return parser


def basis_kwargs(args):
    r"""Basis options for the rational backbones. The regional basis derives
    K from zones * bases_per_zone and fixes its own initialisation, so
    --num_bases and --init do not apply to it."""
    common = dict(degrees=tuple(args.degrees), safe=args.safe, poly=args.poly,
                  init_noise=args.init_noise)
    if args.rational_basis == 'regional':
        return dict(basis='regional', zones=args.zones,
                    bases_per_zone=args.bases_per_zone,
                    zone_overlap=args.zone_overlap, **common)
    return dict(basis=args.rational_basis, init=args.init,
                num_bases=args.num_bases or None, **common)


def freeze_basis(model, args):
    r"""Freezes every basis-shape coefficient, leaving the kernel weights
    trainable. Returns the number of frozen parameters (0 when not asked
    for)."""
    if not getattr(args, 'freeze_basis', False):
        return 0
    frozen = 0
    for name, p in model.named_parameters():
        if name.endswith(('numerator', 'denominator')):
            p.requires_grad_(False)
            frozen += p.numel()
    return frozen


def make_backbone(args, in_channels, out_channels, dim, num_layers, cat,
                  dropout):
    if args.backbone == 'spline':
        return SplineCNN(in_channels, out_channels, dim, num_layers, cat=cat,
                         dropout=dropout, kernel_size=args.kernel_size)
    return RationalCNN(in_channels, out_channels, dim, num_layers, cat=cat,
                       dropout=dropout, kernel_size=args.kernel_size,
                       vp=args.vp, **basis_kwargs(args))


def make_conv(args, in_channels, out_channels, dim, aggr='mean', **kwargs):
    r"""A single continuous-kernel convolution layer (B-spline or rational
    basis) configured from the command line, for models that stack their
    own layers (e.g. examples/faust.py)."""
    if args.backbone == 'spline':
        return BSplineConv(in_channels, out_channels, dim, args.kernel_size,
                           aggr=aggr, pyg_init=args.pyg_init, **kwargs)
    return RationalConv(in_channels, out_channels, dim, args.kernel_size,
                        aggr=aggr, vp=args.vp, pyg_init=args.pyg_init,
                        **basis_kwargs(args), **kwargs)


def make_optimizer(model, args):
    r"""Adam with an optional L2 penalty (:obj:`--basis_wd`) on the rational
    basis coefficients only."""
    wd = getattr(args, 'basis_wd', 0.0)
    trainable = lambda p: p.requires_grad  # --freeze_basis switches some off  # noqa
    if wd <= 0:
        return torch.optim.Adam([p for p in model.parameters() if trainable(p)],
                                lr=args.lr)
    is_basis = lambda n: n.endswith('numerator') or n.endswith('denominator')  # noqa
    basis = [p for n, p in model.named_parameters() if is_basis(n) and trainable(p)]
    rest = [p for n, p in model.named_parameters() if not is_basis(n) and trainable(p)]
    return torch.optim.Adam([{'params': rest},
                             {'params': basis, 'weight_decay': wd}],
                            lr=args.lr)


def describe(model):
    num_params = sum(p.numel() for p in model.parameters())
    return f'{model.__class__.__name__} with {num_params} parameters'
