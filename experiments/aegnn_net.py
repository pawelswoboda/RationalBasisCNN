r"""The AEGNN recognition network (Schaefer, Gehrig, Scaramuzza, CVPR 2022;
``aegnn/models/networks/graph_res.py``, MIT licence) with a pluggable
continuous-kernel convolution: PyG's :class:`SplineConv` (the original),
our pure-PyTorch :class:`BSplineConv`, :class:`RationalConv` with a rational
(safe-Padé) or MLP basis, or a PointNet-style convolution (the replacement of
Jeziorek et al., 2023).

Architecture (recognition, k = 2, channels (1, 8, 16, 16, 16, 32, 32, 32)):

    conv1-bn-elu, conv2-bn-elu, [conv3-bn-elu, conv4-bn-elu] + skip,
    conv5-bn-elu, voxel max-pool (16 x 12 px), [conv6-bn-elu, conv7-bn-elu]
    + skip, voxel max-pool to a 4 x 4 grid, linear.

Convolutions have neither root weight nor bias (AEGNN defaults), mean
aggregation, and 3-D pseudo-coordinates (x, y, t) normalised to [0, 1].
"""
import torch
from torch.nn import Linear
from torch.nn.functional import elu
from torch_geometric.data import Data
from torch_geometric.nn import max_pool, max_pool_x
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn import MessagePassing
from torch_geometric.transforms import Cartesian

from rational_cnn import BSplineConv, RationalConv


def voxel_grid(pos, batch, size, num_cells):
    r"""Cluster index of a fixed voxel grid over the image plane (cells of
    ``size`` pixels, ``num_cells`` = (nx, ny) cells per graph), i.e.
    ``batch * nx * ny + iy * nx + ix``. Equivalent to
    :func:`torch_geometric.nn.pool.voxel_grid` when the batch spans the
    whole image, without depending on ``torch_cluster``."""
    nx, ny = num_cells
    idx = (pos[:, :2] / pos.new_tensor(size)).floor().long()
    ix = idx[:, 0].clamp(0, nx - 1)
    iy = idx[:, 1].clamp(0, ny - 1)
    return batch * (nx * ny) + iy * nx + ix


class MaxPooling(torch.nn.Module):
    r"""AEGNN ``MaxPooling``: voxel max-pooling of features and positions
    (cluster mean), coarsened edges, new edge attributes by ``transform``."""
    def __init__(self, size, num_cells, transform=None):
        super().__init__()
        self.size, self.num_cells, self.transform = size, num_cells, transform

    def forward(self, x, pos, batch, edge_index):
        cluster = voxel_grid(pos, batch, self.size, self.num_cells)
        data = Data(x=x, pos=pos, edge_index=edge_index, batch=batch)
        return max_pool(cluster, data=data, transform=self.transform)


class MaxPoolingX(torch.nn.Module):
    r"""AEGNN ``MaxPoolingX``: max over a fixed ``nx * ny`` grid per graph,
    output ``[num_graphs, nx * ny, C]``."""
    def __init__(self, size, num_cells):
        super().__init__()
        self.size, self.num_cells = size, num_cells

    def forward(self, x, pos, batch, num_graphs):
        cluster = voxel_grid(pos, batch, self.size, self.num_cells)
        n = self.num_cells[0] * self.num_cells[1]
        out, _ = max_pool_x(cluster, x, batch, batch_size=num_graphs, size=n)
        return out.view(num_graphs, n * x.size(1))


class PointNetConv(MessagePassing):
    r"""PointNet-style convolution (Qi et al., 2017, as used by Jeziorek et
    al., 2023, in place of SplineConv): ``max_j W [x_j, u_ij]``, computed as
    ``max_j (W_x x_j + W_u u_ij)`` so that only ``E x C_out`` messages are
    materialised."""
    def __init__(self, in_channels, out_channels, dim):
        super().__init__(aggr='max', node_dim=0)
        self.lin_x = Linear(in_channels, out_channels, bias=False)
        self.lin_u = Linear(dim, out_channels, bias=False)
        self.in_channels, self.out_channels, self.dim = in_channels, \
            out_channels, dim

    def forward(self, x, edge_index, pseudo):
        return self.propagate(edge_index, x=self.lin_x(x), pseudo=pseudo)

    def message(self, x_j, pseudo):
        return x_j + self.lin_u(pseudo)

    def __repr__(self):
        return f'PointNetConv({self.in_channels}, {self.out_channels}, ' \
            f'dim={self.dim})'


class PygSplineConv(MessagePassing):
    r"""AEGNN's original operator: PyG's :class:`SplineConv` (degree-1 open
    B-splines, mean aggregation, no root weight / bias) on the fused
    ``torch_spline_conv`` kernels, which never materialise the ``K`` weight
    matrices per edge. Same mathematics and initialisation as our
    :class:`BSplineConv`; kept as the reference implementation."""
    def __init__(self, in_channels, out_channels, dim, kernel_size, degree=1):
        super().__init__(aggr='mean', node_dim=0)
        from torch.nn import Parameter
        from torch_geometric.nn.inits import uniform
        K = kernel_size ** dim
        self.in_channels, self.out_channels, self.dim = in_channels, \
            out_channels, dim
        self.kernel_size, self.degree = kernel_size, degree
        self.weight = Parameter(torch.empty(K, in_channels, out_channels))
        uniform(K * in_channels, self.weight)
        self.register_buffer('ks', torch.tensor([kernel_size] * dim))
        self.register_buffer('is_open', torch.ones(dim, dtype=torch.uint8))

    def forward(self, x, edge_index, pseudo):
        from torch_spline_conv import spline_basis
        basis, weight_index = spline_basis(pseudo, self.ks, self.is_open,
                                           self.degree)
        return self.propagate(edge_index, x=x, basis=basis,
                              weight_index=weight_index)

    def message(self, x_j, basis, weight_index):
        from torch_spline_conv import spline_weighting
        return spline_weighting(x_j, self.weight, basis, weight_index)

    def __repr__(self):
        return f'PygSplineConv({self.in_channels}, {self.out_channels}, ' \
            f'dim={self.dim}, kernel_size={self.kernel_size})'


def make_aegnn_conv(args, in_channels, out_channels, dim=3):
    r"""One convolution layer of the AEGNN network from the command-line
    arguments of :func:`backbones.add_backbone_args` plus ``--backbone
    pyg_spline`` / ``pointnet``."""
    kw = dict(root_weight=False, bias=False)
    if args.backbone == 'pyg_spline':
        return PygSplineConv(in_channels, out_channels, dim, args.kernel_size)
    if args.backbone == 'spline':
        return BSplineConv(in_channels, out_channels, dim, args.kernel_size,
                           aggr='mean', large=True, **kw)
    if args.backbone == 'pointnet':
        return PointNetConv(in_channels, out_channels, dim)
    assert args.backbone == 'rational'
    return RationalConv(in_channels, out_channels, dim, args.kernel_size,
                        aggr='mean', basis=args.rational_basis, vp=args.vp,
                        degrees=tuple(args.degrees), safe=args.safe,
                        poly=args.poly, init=args.init,
                        init_noise=args.init_noise,
                        num_bases=args.num_bases or None, large=True, **kw)


class GraphRes(torch.nn.Module):
    def __init__(self, args, img_shape, num_classes,
                 channels=(1, 8, 16, 16, 16, 32, 32, 32), dim=3,
                 pool_size=(16, 12)):
        super().__init__()
        n = list(channels)
        W, H = img_shape
        conv = lambda i, o: make_aegnn_conv(args, i, o, dim)  # noqa: E731

        self.conv1, self.norm1 = conv(n[0], n[1]), BatchNorm(n[1])
        self.conv2, self.norm2 = conv(n[1], n[2]), BatchNorm(n[2])
        self.conv3, self.norm3 = conv(n[2], n[3]), BatchNorm(n[3])
        self.conv4, self.norm4 = conv(n[3], n[4]), BatchNorm(n[4])
        self.conv5, self.norm5 = conv(n[4], n[5]), BatchNorm(n[5])
        cells5 = (-(-W // pool_size[0]), -(-H // pool_size[1]))
        self.pool5 = MaxPooling(pool_size, cells5,
                                transform=Cartesian(norm=True, cat=False))
        self.conv6, self.norm6 = conv(n[5], n[6]), BatchNorm(n[6])
        self.conv7, self.norm7 = conv(n[6], n[7]), BatchNorm(n[7])
        self.pool7 = MaxPoolingX((W // 4, H // 4), (4, 4))
        self.fc = Linear(n[7] * 16, num_classes, bias=False)

    def forward(self, data):
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        x = self.norm1(elu(self.conv1(x, ei, ea)))
        x = self.norm2(elu(self.conv2(x, ei, ea)))
        x_sc = x
        x = self.norm3(elu(self.conv3(x, ei, ea)))
        x = self.norm4(elu(self.conv4(x, ei, ea)))
        x = x + x_sc
        x = self.norm5(elu(self.conv5(x, ei, ea)))
        data = self.pool5(x, data.pos, data.batch, ei)
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        x_sc = x
        x = self.norm6(elu(self.conv6(x, ei, ea)))
        x = self.norm7(elu(self.conv7(x, ei, ea)))
        x = x + x_sc
        x = self.pool7(x, data.pos, data.batch, data.num_graphs)
        return self.fc(x)
