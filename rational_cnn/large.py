r"""Memory-lean aggregation for continuous-kernel convolutions on very large
graphs (e.g. event-camera graphs with :math:`10^7` edges per batch).

For a dense basis :math:`R \in \mathbb{R}^{E \times K}` the operator

.. math::
    \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
    \Big( \sum_p R_p(\mathbf{u}_{ij}) \mathbf{\Theta}_p \Big) \mathbf{x}_j
    = \sum_p \mathbf{\Theta}_p^{\top} \underbrace{\sum_{j \in \mathcal{N}(i)}
    R_p(\mathbf{u}_{ij}) \mathbf{x}_j}_{\mathbf{Y}_{i,p}}

is evaluated *basis first*: the aggregated features
:math:`\mathbf{Y} \in \mathbb{R}^{N \times K \times C_{in}}` are formed with
one scatter per basis function and then multiplied by the stacked weights.
Peak memory is :math:`O(E \cdot C_{in})` (one transient message tensor at a
time) plus :math:`O(N K C_{in})`, instead of the :math:`O(E K C)` of the
straightforward per-edge kernel evaluation, and only the basis values
(:math:`E \times K`) and node features are kept for the backward pass.
"""
import torch


class _BasisAggregate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, R, x, row, col, N):
        # R: [E, K] basis values (already scaled for mean aggregation),
        # x: [N, C], row: source j, col: target i.
        E, K = R.shape
        C = x.size(1)
        Y = x.new_zeros(N, K, C)
        for p in range(K):
            Y[:, p].index_add_(0, col, R[:, p:p + 1] * x[row])
        ctx.save_for_backward(R, x, row, col)
        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        R, x, row, col = ctx.saved_tensors
        E, K = R.shape
        grad_R = R.new_empty(E, K) if ctx.needs_input_grad[0] else None
        grad_x = torch.zeros_like(x) if ctx.needs_input_grad[1] else None
        x_row = x[row]
        for p in range(K):
            g = grad_Y[:, p][col]  # [E, C]
            if grad_R is not None:
                grad_R[:, p] = (g * x_row).sum(-1)
            if grad_x is not None:
                grad_x.index_add_(0, row, R[:, p:p + 1] * g)
        return grad_R, grad_x, None, None, None


def basis_aggregate(R, x, edge_index, num_nodes, aggr='mean'):
    r"""Returns :math:`\mathbf{Y}_{i,p} = \square_{j \in \mathcal{N}(i)}
    R_p(\mathbf{u}_{ij}) \mathbf{x}_j` of shape :obj:`[N, K, C_in]` for
    :obj:`aggr` in :obj:`'add'` / :obj:`'mean'` (PyG :obj:`source_to_target`
    convention, :obj:`edge_index[0]` = source :math:`j`)."""
    assert aggr in ['add', 'mean']
    row, col = edge_index[0], edge_index[1]
    if aggr == 'mean':
        deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        deg.index_add_(0, col, torch.ones_like(col, dtype=x.dtype))
        R = R / deg.clamp(min=1)[col].unsqueeze(-1)
    return _BasisAggregate.apply(R.contiguous(), x.contiguous(), row, col,
                                 num_nodes)


def basis_conv(R, x, edge_index, weight, aggr='mean'):
    r"""The full convolution :math:`\sum_p \mathbf{\Theta}_p^{\top}
    \mathbf{Y}_{i,p}` for :obj:`weight` of shape :obj:`[K, C_in, C_out]`."""
    N = x.size(0)
    K, C_in, C_out = weight.shape
    Y = basis_aggregate(R, x, edge_index, N, aggr=aggr)
    return Y.view(N, K * C_in) @ weight.view(K * C_in, C_out)


def chunked_basis(basis, pseudo, chunk_size=2**18):
    r"""Evaluates a basis module :obj:`[E, D] -> [E, K]` in chunks of
    :obj:`chunk_size` edges with activation checkpointing, so that the
    intermediate polynomial / MLP features (up to hundreds per edge) are
    never stored for the backward pass; only the basis values :obj:`[E, K]`
    are."""
    E = pseudo.size(0)
    if E <= chunk_size:
        return basis(pseudo)
    from torch.utils.checkpoint import checkpoint
    return torch.cat([checkpoint(basis, pseudo[s:s + chunk_size],
                                 use_reentrant=False)
                      for s in range(0, E, chunk_size)], dim=0)
