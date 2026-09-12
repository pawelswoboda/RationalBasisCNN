import torch
from rational_cnn import BSplineConv
from rational_cnn.bspline import open_bspline_basis_1d, bspline_basis


def test_bspline_basis_partition_of_unity():
    u = torch.rand(100, 3)
    u[0] = 0
    u[1] = 1
    for K in [2, 3, 5]:
        b = open_bspline_basis_1d(u[:, 0], K)
        assert b.size() == (100, K)
        assert (b >= 0).all()
        assert torch.allclose(b.sum(-1), torch.ones(100))

        idx, weight = bspline_basis(u, [K, K, K])
        assert idx.size() == (100, 8) and weight.size() == (100, 8)
        assert idx.min() >= 0 and idx.max() < K**3
        assert torch.allclose(weight.sum(-1), torch.ones(100))

    # Index ordering: first dimension fastest.
    u = torch.tensor([[0.0, 1.0]])
    idx, weight = bspline_basis(u, [3, 4])
    dense = torch.zeros(1, 12).scatter_add_(1, idx, weight)
    assert dense[0, 0 + 3 * 3] == 1


def test_bspline_conv_matches_naive():
    torch.manual_seed(0)
    N, E, cin, cout, K = 10, 40, 3, 4, 5
    x = torch.randn(N, cin, dtype=torch.double)
    edge_index = torch.randint(0, N, (2, E))
    pseudo = torch.rand(E, 2, dtype=torch.double)
    conv = BSplineConv(cin, cout, dim=2, kernel_size=K).double()
    assert conv.__repr__() == 'BSplineConv(3, 4, dim=2, kernel_size=[5, 5])'
    out = conv(x, edge_index, pseudo)

    idx, weight = bspline_basis(pseudo, [K, K])
    B = torch.zeros(E, K * K, dtype=torch.double).scatter_add_(1, idx, weight)
    ref = torch.zeros(N, cout, dtype=torch.double)
    cnt = torch.zeros(N, dtype=torch.double)
    for e in range(E):
        j, i = edge_index[0, e], edge_index[1, e]
        ref[i] += x[j] @ torch.einsum('p,pio->io', B[e], conv.weight)
        cnt[i] += 1
    ref = ref / cnt.clamp(min=1).view(-1, 1) + x @ conv.root + conv.bias
    assert torch.allclose(out, ref)

    x.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: conv(x, edge_index, pseudo),
                                    (x, ))


def test_bspline_conv_options():
    conv = BSplineConv(3, 4, dim=1, kernel_size=3, root_weight=False,
                       bias=False)
    assert conv.root is None and conv.bias is None
    x = torch.randn(10, 3)
    edge_index = torch.randint(0, 10, (2, 30))
    out = conv(x, edge_index, torch.rand(30, 1))
    assert out.size() == (10, 4)
    conv.reset_parameters()
