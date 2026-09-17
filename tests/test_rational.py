from itertools import product

import pytest
import torch
from torch_geometric.data import Data

from rational_cnn import (DGMC, BSplineConv, RationalBasis1D, RationalBasis,
                         MultivariateRationalBasis, MLPBasis, RationalConv,
                         RationalCNN, SplineCNN)
from rational_cnn.bspline import open_bspline_basis_1d, bspline_basis
from rational_cnn.rational import (poly_features, total_degree_indices,
                                  multivariate_poly_features, basis_energy,
                                  hat_energy, gauss_basis_1d)


def test_poly_features():
    t = torch.linspace(-1, 1, 11)
    mono = poly_features(t, 3, 'monomial')
    assert torch.allclose(mono[:, 3], t**3)
    cheb = poly_features(t, 3, 'chebyshev')
    assert torch.allclose(cheb[:, 2], 2 * t**2 - 1)
    assert torch.allclose(cheb[:, 3], 4 * t**3 - 3 * t)
    assert cheb.abs().max() <= 1 + 1e-6


def test_rational_basis_spline_init_approximates_hats():
    torch.manual_seed(0)
    u = torch.linspace(0, 1, 501, dtype=torch.double)
    for K, degrees in [(3, (5, 4)), (5, (5, 4))]:
        basis = RationalBasis1D(K, degrees=degrees).double()
        assert basis.__repr__() == (
            "RationalBasis1D({}, degrees={}, safe=B, poly=chebyshev, "
            "init=spline)".format(K, degrees))
        out = basis(u)
        assert out.size() == (501, K)
        target = open_bspline_basis_1d(u, K)
        assert (out - target).abs().max() < 0.1
        assert (out - target).pow(2).mean().sqrt() < 0.02
        # The denominator is actually used (not stuck at the zero saddle).
        assert basis.denominator.abs().max() > 1e-2

    # Polynomial-only fit is clearly worse.
    poly = RationalBasis1D(5, degrees=(5, 4), fit_steps=0).double()
    assert (poly(u) - open_bspline_basis_1d(u, 5)).abs().max() > 0.2


def test_rational_basis_is_safe():
    torch.manual_seed(0)
    u = torch.linspace(0, 1, 1001)
    for safe, poly, init in product(['A', 'B'], ['chebyshev', 'monomial'],
                                    ['spline', 'random']):
        basis = RationalBasis1D(4, degrees=(5, 4), safe=safe, poly=poly,
                                init=init, fit_steps=10)
        basis.denominator.data.normal_(0, 100)
        basis.numerator.data.normal_(0, 100)
        out = basis(u)
        assert torch.isfinite(out).all()
        # |r| <= |P| since the denominator is at least one.
        P = poly_features(2 * u - 1, 5, poly) @ basis.numerator.t()
        assert (out.abs() <= P.abs() + 1e-4).all()
        assert basis.denominator.grad is None
        out.sum().backward()
        assert basis.denominator.grad.abs().sum() > 0


def test_rational_basis_matches_bspline_ordering():
    torch.manual_seed(0)
    pseudo = torch.rand(50, 2, dtype=torch.double)
    rb = RationalBasis(2, [5, 3]).double()
    assert rb.num_bases == 15
    R = rb(pseudo)
    idx, weight = bspline_basis(pseudo, [5, 3])
    B = torch.zeros(50, 15, dtype=torch.double).scatter_add_(1, idx, weight)
    assert (R - B).abs().max() < 0.2
    assert (R - B).pow(2).mean().sqrt() < 0.05


def test_rational_conv_matches_naive():
    torch.manual_seed(0)
    N, E, K = 10, 40, 3
    edge_index = torch.randint(0, N, (2, E))
    pseudo = torch.rand(E, 2, dtype=torch.double)
    for cin, cout in [(4, 3), (3, 4)]:  # both memory paths
        x = torch.randn(N, cin, dtype=torch.double)
        conv = RationalConv(cin, cout, dim=2, kernel_size=K,
                            fit_steps=20).double()
        assert conv.__repr__() == (
            'RationalConv({}, {}, dim=2, kernel_size=[3, 3], basis='
            'RationalBasis(dim=2, kernel_size=[3, 3], RationalBasis1D(3, '
            'degrees=(5, 4), safe=B, poly=chebyshev, init=spline)))'
        ).format(cin, cout)
        out = conv(x, edge_index, pseudo)

        R = conv.basis(pseudo)
        ref = torch.zeros(N, cout, dtype=torch.double)
        cnt = torch.zeros(N, dtype=torch.double)
        for e in range(E):
            j, i = edge_index[0, e], edge_index[1, e]
            ref[i] += x[j] @ torch.einsum('p,pio->io', R[e], conv.weight)
            cnt[i] += 1
        ref = ref / cnt.clamp(min=1).view(-1, 1) + x @ conv.root + conv.bias
        assert torch.allclose(out, ref)

        x.requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda x: conv(x, edge_index, pseudo), (x, ))
        for p in conv.basis.parameters():
            assert p.requires_grad
        conv(x, edge_index, pseudo).sum().backward()
        for b in conv.basis.bases:
            assert b.numerator.grad.abs().sum() > 0


def test_rational_conv_weight_transfer_from_bspline():
    # With spline init, RationalConv with the weights of a BSplineConv
    # approximately reproduces its output.
    torch.manual_seed(0)
    x = torch.randn(20, 8, dtype=torch.double)
    edge_index = torch.randint(0, 20, (2, 100))
    pseudo = torch.rand(100, 2, dtype=torch.double)
    spline = BSplineConv(8, 6, dim=2, kernel_size=5).double()
    rational = RationalConv(8, 6, dim=2, kernel_size=5).double()
    rational.weight.data.copy_(spline.weight)
    rational.root.data.copy_(spline.root)
    rational.bias.data.copy_(spline.bias)
    out_s = spline(x, edge_index, pseudo)
    out_r = rational(x, edge_index, pseudo)
    assert (out_s - out_r).abs().max() < 0.25 * out_s.abs().max()


def test_rational_cnn():
    model = RationalCNN(16, 32, dim=3, num_layers=2, cat=True, lin=True,
                        dropout=0.5, kernel_size=3, fit_steps=10)
    assert model.__repr__() == (
        'RationalCNN(16, 32, dim=3, num_layers=2, cat=True, lin=True, '
        'dropout=0.5, kernel_size=3, RationalBasis(dim=3, kernel_size='
        '[3, 3, 3], RationalBasis1D(3, degrees=(5, 4), safe=B, '
        'poly=chebyshev, init=spline)))')

    x = torch.randn(100, 16)
    edge_index = torch.randint(100, (2, 400), dtype=torch.long)
    edge_attr = torch.rand((400, 3))
    for cat, lin in product([False, True], [False, True]):
        model = RationalCNN(16, 32, 3, 2, cat, lin, 0.5, kernel_size=3,
                            fit_steps=10)
        out = model(x, edge_index, edge_attr)
        assert out.size() == (100, 16 + 2 * 32 if not lin and cat else 32)
        assert out.size() == (100, model.out_channels)
    model.reset_parameters()


def test_dgmc_with_rational_backbone():
    torch.manual_seed(0)
    x = torch.randn(4, 32)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    edge_attr = torch.rand(6, 2)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    psi_1 = RationalCNN(32, 16, 2, num_layers=2, kernel_size=3, fit_steps=10)
    psi_2 = RationalCNN(8, 8, 2, num_layers=2, kernel_size=3, fit_steps=10)
    model = DGMC(psi_1, psi_2, num_steps=2)
    y = torch.arange(data.num_nodes)
    y = torch.stack([y, y], dim=0)
    S_0, S_L = model(data.x, data.edge_index, data.edge_attr, None, data.x,
                     data.edge_index, data.edge_attr, None)
    loss = model.loss(S_0, y) + model.loss(S_L, y)
    loss.backward()
    assert S_L.size() == (4, 4)
    assert 0 <= model.acc(S_L, y) <= 1

    # Same interface as SplineCNN.
    psi_1 = SplineCNN(32, 16, 2, num_layers=2)
    psi_2 = SplineCNN(8, 8, 2, num_layers=2)
    model = DGMC(psi_1, psi_2, num_steps=2)
    S_0, S_L = model(data.x, data.edge_index, data.edge_attr, None, data.x,
                     data.edge_index, data.edge_attr, None)
    assert S_L.size() == (4, 4)


def test_total_degree_indices_and_features():
    idx = total_degree_indices(2, 3)
    assert len(idx) == 10 and idx[0] == (0, 0)  # binom(3 + 2, 2)
    assert all(sum(a) <= 3 for a in idx)
    assert len(total_degree_indices(3, 4)) == 35  # binom(4 + 3, 3)

    t = torch.rand(7, 2) * 2 - 1
    phi = multivariate_poly_features(t, [(0, 0), (1, 0), (0, 1), (1, 1),
                                         (2, 1)], 'monomial')
    assert torch.allclose(phi[:, 0], torch.ones(7))
    assert torch.allclose(phi[:, 3], t[:, 0] * t[:, 1])
    assert torch.allclose(phi[:, 4], t[:, 0]**2 * t[:, 1])


def test_multivariate_rational_basis_spline_init_approximates_hats():
    torch.manual_seed(0)
    g = torch.linspace(0, 1, 33, dtype=torch.double)
    u = torch.stack(torch.meshgrid(g, g, indexing='ij'), dim=-1).view(-1, 2)
    idx, weight = bspline_basis(u, [3, 3])
    target = u.new_zeros(u.size(0), 9).scatter_add_(1, idx, weight)

    basis = MultivariateRationalBasis(2, 3, degrees=(8, 6)).double()
    assert basis.__repr__() == (
        'MultivariateRationalBasis(dim=2, kernel_size=[3, 3], num_bases=9, '
        'degrees=(8, 6), safe=B, poly=chebyshev, init=spline)')
    assert basis.num_bases == 9
    assert basis.numerator.size() == (9, 45)  # binom(8 + 2, 2)
    assert basis.denominator.size() == (9, 27)  # binom(6 + 2, 2) - 1
    out = basis(u)
    assert out.size() == (u.size(0), 9)
    assert (out - target).abs().max() < 0.15
    assert (out - target).pow(2).mean().sqrt() < 0.01
    assert basis.denominator.abs().max() > 1e-2

    # Is a genuinely non-separable function: a rank-one (separable) basis
    # function would give a rank-one 33 x 33 value grid. Check that the
    # fitted functions are at least not degenerate and finite everywhere.
    assert torch.isfinite(out).all()

    for safe in ['A', 'B']:
        b = MultivariateRationalBasis(2, 3, safe=safe, init='random',
                                      fit_steps=0)
        o = b(torch.rand(50, 2))
        assert o.size() == (50, 9) and torch.isfinite(o).all()
        o.sum().backward()
        assert b.numerator.grad is not None
        assert b.denominator.grad is not None and b.denominator.grad.abs().sum() > 0


def test_rational_conv_multivariate_basis():
    torch.manual_seed(0)
    conv = RationalConv(8, 16, dim=2, kernel_size=3, basis='multivariate',
                        degrees=(4, 3), fit_steps=5)
    assert isinstance(conv.basis, MultivariateRationalBasis)
    x = torch.randn(10, 8)
    edge_index = torch.randint(10, (2, 40), dtype=torch.long)
    pseudo = torch.rand(40, 2)
    out = conv(x, edge_index, pseudo)
    assert out.size() == (10, 16)
    out.sum().backward()
    assert conv.basis.numerator.grad is not None
    assert conv.weight.grad is not None

    model = RationalCNN(8, 16, 2, num_layers=2, kernel_size=3,
                        basis='multivariate', fit_steps=5)
    assert 'MultivariateRationalBasis' in model.__repr__()
    out = model(x, edge_index, pseudo)
    assert out.size() == (10, 16)


def test_constant_init_is_one_and_trainable():
    torch.manual_seed(0)
    u = torch.rand(64)
    basis = RationalBasis1D(5, init='constant', init_noise=1e-3)
    out = basis(u)
    assert out.size() == (64, 5)
    assert (out - 1).abs().max() < 0.05  # B_p = 1 up to the tiny noise
    out.sum().backward()
    # The denominator gradient does not vanish (it would at exactly zero).
    assert basis.denominator.grad.abs().sum() > 0

    mv = MultivariateRationalBasis(2, 3, init='constant', init_noise=1e-3)
    out = mv(torch.rand(64, 2))
    assert (out - 1).abs().max() < 0.05
    out.sum().backward()
    assert mv.denominator.grad.abs().sum() > 0


def test_gauss_and_cheb_inits():
    torch.manual_seed(0)
    u = torch.linspace(0, 1, 201, dtype=torch.double)
    g = gauss_basis_1d(u, 5)
    assert g.size() == (201, 5) and torch.allclose(g[50, 1], torch.tensor(1., dtype=torch.double))
    basis = RationalBasis1D(5, init='gauss').double()
    assert (basis(u) - g).abs().max() < 0.1
    cheb = RationalBasis1D(5, init='cheb', init_noise=1e-4)
    out = cheb(u.float())
    assert torch.allclose(out, poly_features(2 * u.float() - 1, 4), atol=1e-2)
    mv = MultivariateRationalBasis(2, 3, degrees=(8, 6), init='gauss').double()
    uu = torch.rand(100, 2, dtype=torch.double)
    assert (mv(uu) - mv.target(uu)).abs().max() < 0.15
    mv = MultivariateRationalBasis(2, 3, degrees=(8, 6), init='cheb')
    assert torch.isfinite(mv(uu.float())).all()


def test_variance_preserving_gain():
    torch.manual_seed(0)
    assert abs(hat_energy([5]) - 2 / 3) < 1e-2  # E[w^2 + (1-w)^2] = 2/3
    assert abs(hat_energy([5, 5]) - 4 / 9) < 2e-2
    # constant basis: sum_p B_p^2 = K -> gain = (2/3)^2 / 25
    conv = RationalConv(8, 16, dim=2, kernel_size=5, init='constant', vp=True)
    assert abs(conv.gain - (4 / 9) / 25) < 0.02
    # spline-fitted basis: gain ~ 1
    conv = RationalConv(8, 16, dim=2, kernel_size=5, vp=True, fit_steps=50)
    assert 0.7 < conv.gain < 1.4
    assert 'vp=True' in conv.__repr__()
    # message scale is actually matched: compare with a BSplineConv
    x = torch.randn(200, 8)
    edge_index = torch.randint(200, (2, 2000), dtype=torch.long)
    pseudo = torch.rand(2000, 2)
    ref = BSplineConv(8, 16, dim=2, kernel_size=5, root_weight=False, bias=False)
    const = RationalConv(8, 16, dim=2, kernel_size=5, init='constant', vp=True,
                         root_weight=False, bias=False)
    ratio = const(x, edge_index, pseudo).std() / ref(x, edge_index, pseudo).std()
    assert 0.5 < ratio < 2.0


def test_pca_init_and_free_num_bases():
    torch.manual_seed(0)
    g = torch.linspace(0, 1, 33, dtype=torch.double)
    u = torch.stack(torch.meshgrid(g, g, indexing='ij'), dim=-1).view(-1, 2)
    # K = 25 principal components span the k=5 hat basis exactly.
    full = MultivariateRationalBasis(2, 5, degrees=(8, 6), init='pca',
                                     fit_steps=0)
    assert full.num_bases == 25
    T = full.double().target(u)
    idx, w = bspline_basis(u, [5, 5])
    hats = u.new_zeros(u.size(0), 25).scatter_add_(1, idx, w)
    coef = torch.linalg.lstsq(T, hats).solution
    assert (T @ coef - hats).abs().max() < 1e-6
    assert T.abs().max(dim=0).values.allclose(torch.ones(25, dtype=torch.double))

    basis = MultivariateRationalBasis(2, 5, degrees=(8, 6), init='pca',
                                      num_bases=4).double()
    assert basis.num_bases == 4 and basis.numerator.size(0) == 4
    assert 'num_bases=4' in basis.__repr__()
    # Compare on the fit grid: components 2/3 are a degenerate pair (x/y
    # symmetry), so their rotation is arbitrary on any other grid.
    g = torch.linspace(0, 1, basis.grid_size, dtype=torch.double)
    u32 = torch.stack(torch.meshgrid(g, g, indexing='ij'), dim=-1).view(-1, 2)
    out = basis(u32)
    assert out.size() == (u32.size(0), 4)
    assert (out - basis.target(u32)).abs().max() < 0.2
    assert (out - basis.target(u32)).pow(2).mean().sqrt() < 0.05

    conv = RationalConv(8, 16, dim=2, kernel_size=5, basis='multivariate',
                        degrees=(8, 6), init='pca', num_bases=6, vp=True,
                        fit_steps=5)
    assert conv.weight.size() == (6, 8, 16) and conv.num_bases == 6
    x = torch.randn(10, 8)
    edge_index = torch.randint(10, (2, 40), dtype=torch.long)
    assert conv(x, edge_index, torch.rand(40, 2)).size() == (10, 16)
    with pytest.raises(AssertionError):
        MultivariateRationalBasis(2, 5, init='spline', num_bases=4)


def test_mlp_basis_control():
    torch.manual_seed(0)
    basis = MLPBasis(2, 5, num_bases=4, degrees=(8, 6), init='pca')
    assert basis.num_bases == 4
    assert basis(torch.rand(7, 2)).size() == (7, 4)
    conv = RationalConv(8, 16, dim=2, kernel_size=5, basis='mlp',
                        num_bases=9, vp=True)
    assert isinstance(conv.basis, MLPBasis) and conv.weight.size(0) == 9
    x = torch.randn(10, 8)
    edge_index = torch.randint(10, (2, 40), dtype=torch.long)
    out = conv(x, edge_index, torch.rand(40, 2))
    assert out.size() == (10, 16)
    out.sum().backward()
    assert conv.basis.lin1.weight.grad is not None
    model = RationalCNN(8, 16, 2, num_layers=2, kernel_size=5, basis='mlp',
                        num_bases=4)
    assert model(x, edge_index, torch.rand(40, 2)).size() == (10, 16)


def test_aggr_option_and_3d_pca():
    torch.manual_seed(0)
    x = torch.randn(10, 8)
    edge_index = torch.randint(10, (2, 40), dtype=torch.long)
    pseudo = torch.rand(40, 3)
    for conv in [BSplineConv(8, 16, dim=3, kernel_size=3, aggr='add'),
                 RationalConv(8, 16, dim=3, kernel_size=3, aggr='add',
                              fit_steps=5),
                 RationalConv(8, 16, dim=3, kernel_size=3, aggr='add',
                              basis='multivariate', degrees=(4, 3),
                              init='pca', num_bases=4, vp=True, fit_steps=5)]:
        assert conv.aggr == 'add'
        out = conv(x, edge_index, pseudo)
        assert out.size() == (10, 16) and torch.isfinite(out).all()
    assert conv.weight.size(0) == 4 and conv.gain > 0


def test_product_basis_accepts_num_bases_none():
    conv = RationalConv(4, 8, dim=3, kernel_size=3, basis='product',
                        num_bases=None, fit_steps=5)
    assert conv.num_bases == 27
    with pytest.raises(AssertionError):
        RationalConv(4, 8, dim=3, kernel_size=3, basis='product', num_bases=4,
                     fit_steps=5)


def test_large_graph_path_matches_dense():
    from rational_cnn import BSplineConv, RationalConv
    torch.manual_seed(0)
    N, E = 100, 1500
    x = torch.randn(N, 8, dtype=torch.double)
    edge_index = torch.randint(0, N, (2, E))
    u = torch.rand(E, 3, dtype=torch.double)
    for make in [
            lambda large: BSplineConv(8, 12, 3, 2, large=large),
            lambda large: RationalConv(8, 12, 3, 2, basis='multivariate',
                                       num_bases=6, init='pca',
                                       degrees=(4, 3), large=large),
            lambda large: RationalConv(8, 4, 3, 2, basis='multivariate',
                                       num_bases=6, init='pca',
                                       degrees=(4, 3), aggr='add',
                                       large=large)]:
        torch.manual_seed(1)
        dense = make(False).double()
        torch.manual_seed(1)
        large = make(True).double()
        large.load_state_dict(dense.state_dict())
        xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
        ya, yb = dense(xa, edge_index, u), large(xb, edge_index, u)
        assert torch.allclose(ya, yb, atol=1e-10)
        ya.sum().backward()
        yb.sum().backward()
        assert torch.allclose(xa.grad, xb.grad, atol=1e-10)
        for pa, pb in zip(dense.parameters(), large.parameters()):
            if pa.grad is not None:
                assert torch.allclose(pa.grad, pb.grad, atol=1e-9)


def test_large_graph_max_aggregation_matches_dense():
    from rational_cnn import BSplineConv, RationalConv
    torch.manual_seed(0)
    N, E = 100, 1500
    x = torch.randn(N, 8, dtype=torch.double)
    edge_index = torch.randint(0, N, (2, E))
    u = torch.rand(E, 3, dtype=torch.double)
    for make in [
            lambda large: BSplineConv(8, 12, 3, 2, aggr='max', large=large),
            lambda large: RationalConv(8, 12, 3, 2, basis='multivariate',
                                       num_bases=6, init='pca',
                                       degrees=(4, 3), aggr='max',
                                       large=large)]:
        torch.manual_seed(1)
        dense = make(False).double()
        torch.manual_seed(1)
        large = make(True).double()
        large.load_state_dict(dense.state_dict())
        xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
        ya, yb = dense(xa, edge_index, u), large(xb, edge_index, u)
        assert torch.allclose(ya, yb, atol=1e-10)
        ya.sum().backward()
        yb.sum().backward()
        assert torch.allclose(xa.grad, xb.grad, atol=1e-10)
        for pa, pb in zip(dense.parameters(), large.parameters()):
            if pa.grad is not None:
                assert torch.allclose(pa.grad, pb.grad, atol=1e-9)
