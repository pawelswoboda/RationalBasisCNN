import torch
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import uniform


def open_bspline_1d(u, kernel_size):
    r"""Degree-1 open B-spline basis over :obj:`kernel_size` uniformly spaced
    knots on :math:`[0, 1]`, following the semantics of
    :obj:`torch_spline_conv`.

    Returns the two knot indices and the two (hat) weights per entry, i.e.
    tensors of shape :obj:`[..., 2]`.
    """
    v = u.clamp(0, 1) * (kernel_size - 1)
    k = v.floor().clamp(max=kernel_size - 2).long()
    frac = v - k.to(u.dtype)
    idx = torch.stack([k, k + 1], dim=-1)
    weight = torch.stack([1 - frac, frac], dim=-1)
    return idx, weight


def open_bspline_basis_1d(u, kernel_size):
    r"""Dense degree-1 open B-spline basis of shape :obj:`[..., kernel_size]`
    (the classic hat functions, forming a partition of unity)."""
    idx, weight = open_bspline_1d(u, kernel_size)
    out = u.new_zeros(u.size() + (kernel_size, ))
    return out.scatter_add_(-1, idx, weight)


def bspline_basis(pseudo, kernel_size):
    r"""Tensor-product degree-1 open B-spline basis for pseudo-coordinates
    :obj:`pseudo` of shape :obj:`[E, D]`.

    Returns the :obj:`2**D` non-zero weight indices (in
    :obj:`[0, prod(kernel_size))`) and their values, both of shape
    :obj:`[E, 2**D]`.
    """
    E, D = pseudo.size()
    idx, weight = pseudo.new_zeros(E, 1).long(), pseudo.new_ones(E, 1)
    offset = 1
    for d in range(D):
        idx_d, weight_d = open_bspline_1d(pseudo[:, d], kernel_size[d])
        idx = (idx.unsqueeze(-1) + offset * idx_d.unsqueeze(-2)).view(E, -1)
        weight = (weight.unsqueeze(-1) * weight_d.unsqueeze(-2)).view(E, -1)
        offset *= kernel_size[d]
    return idx, weight


class BSplineConv(MessagePassing):
    r"""Pure-PyTorch re-implementation of the spline-based convolutional
    operator from `"SplineCNN: Fast Geometric Deep Learning with Continuous
    B-Spline Kernels" <https://arxiv.org/abs/1711.08920>`_ (degree-1 open
    B-splines, mean aggregation, root weight and bias), *i.e.*

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\mathrm{root}} \mathbf{x}_i +
        \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)}
        \Big( \sum_p B_p(\mathbf{u}_{ij}) \mathbf{\Theta}_p \Big) \mathbf{x}_j
        + \mathbf{b},

    where :math:`B_p` is the tensor product of hat functions over
    :obj:`kernel_size` knots per pseudo-coordinate dimension. It does not
    depend on :obj:`torch_spline_conv` or :obj:`pyg-lib`.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        dim (int): Pseudo-coordinate dimensionality.
        kernel_size (int or [int]): Number of knots per dimension.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not
            learn an additive bias. (default: :obj:`True`)
    """
    def __init__(self, in_channels, out_channels, dim, kernel_size,
                 root_weight=True, bias=True, aggr='mean',
                 pyg_init=False):
        super(BSplineConv, self).__init__(aggr=aggr, node_dim=0)
        self.pyg_init = pyg_init

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        assert len(kernel_size) == dim
        self.kernel_size = list(kernel_size)

        K = 1
        for k in self.kernel_size:
            K *= k
        self.num_bases = K

        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))
        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0) * self.weight.size(1)
        uniform(size, self.weight)
        if self.pyg_init:
            # torch_geometric.nn.SplineConv: root = Linear(uniform init,
            # bound 1/sqrt(in_channels)), bias = 0.
            uniform(self.in_channels, self.root)
            if self.bias is not None:
                self.bias.data.zero_()
        else:
            uniform(size, self.root)
            uniform(size, self.bias)

    def basis(self, pseudo):
        r"""Sparse basis: non-zero weight indices and values, each of shape
        :obj:`[E, 2**dim]`."""
        return bspline_basis(pseudo, self.kernel_size)

    def forward(self, x, edge_index, pseudo):
        """"""
        N, K, C_out = x.size(0), self.num_bases, self.out_channels
        # Transform node features by all K weight matrices at once; the
        # per-edge kernel is then a sparse combination of these.
        xw = x @ self.weight.permute(1, 0, 2).reshape(x.size(1), -1)
        xw = xw.view(N, K, C_out)
        idx, weight = self.basis(pseudo)

        out = self.propagate(edge_index, xw=xw, idx=idx, weight=weight)

        if self.root is not None:
            out = out + x @ self.root
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, xw_j, idx, weight):
        # xw_j: [E, K, C_out], idx/weight: [E, S] with S = 2**dim.
        idx = idx.unsqueeze(-1).expand(-1, -1, xw_j.size(-1))
        msg = torch.gather(xw_j, 1, idx)
        return (msg * weight.unsqueeze(-1)).sum(dim=1)

    def __repr__(self):
        return '{}({}, {}, dim={}, kernel_size={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.dim, self.kernel_size)
