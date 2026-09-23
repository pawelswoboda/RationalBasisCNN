import argparse
import sys
import os.path as osp

import pytest
import torch

sys.path.insert(0, osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))),
                            'experiments'))
from rational_cnn import DGMC, RationalConv, RegionalRationalBasis  # noqa: E402
from rational_cnn.rational import DEFAULT_ZONE_RADII, wendland_c2  # noqa: E402
from backbones import (add_backbone_args, freeze_basis,  # noqa: E402
                       make_backbone, make_optimizer)


def grid(n=151, dim=2):
    axes = [torch.linspace(0, 1, n)] * dim
    return torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).view(-1, dim)


def args_for(flags):
    p = argparse.ArgumentParser()
    add_backbone_args(p)
    return p.parse_args(flags.split())


def test_wendland_is_compactly_supported():
    t = torch.tensor([0.0, 0.5, 0.999, 1.0, 1.5, 10.0])
    v = wendland_c2(t)
    assert v[0] == 1.0
    assert (v[:3] > 0).all()
    assert (v[3:] == 0).all()          # exactly zero past the support


@pytest.mark.parametrize('zones,per_zone', [(4, 1), (3, 3), (2, 2)])
def test_num_bases_is_zones_times_bases_per_zone(zones, per_zone):
    b = RegionalRationalBasis(2, 5, zones=zones, bases_per_zone=per_zone,
                              degrees=(8, 6))
    assert b.num_bases == zones * per_zone
    assert b(grid(31)).shape == (31 * 31, zones * per_zone)


def test_basis_is_exactly_zero_outside_its_zone():
    """The property the experiment exists to test: a rational can never vanish
    on an open set, but windowed by a compactly supported bump it must."""
    zones, per_zone = 4, 1
    b = RegionalRationalBasis(2, 5, zones=zones, bases_per_zone=per_zone,
                              degrees=(8, 6))
    # push the coefficients far from init, so this is not an artefact of a
    # near-constant rational
    with torch.no_grad():
        b.numerator.normal_(0, 3.0)
        b.denominator.normal_(0, 3.0)
    u = grid(201)
    with torch.no_grad():
        B = b(u)
    rho = (u - 0.5).norm(dim=-1)
    for j in range(b.num_bases):
        m = j // per_zone
        lo = float(b.zone_centre[m] - b.zone_half[m])
        hi = float(b.zone_centre[m] + b.zone_half[m])
        outside = (rho < lo) | (rho > hi)
        if outside.any():
            assert float(B[outside, j].abs().max()) == 0.0
    # and it is genuinely sparse: far fewer than K active at a point
    active = (B.abs() > 0).float().sum(-1).mean()
    assert float(active) < b.num_bases


def test_windows_are_a_partition_of_unity():
    b = RegionalRationalBasis(2, 5, zones=4, bases_per_zone=1, degrees=(8, 6))
    w = b.windows(grid(101))
    assert w.shape[-1] == 4
    assert float(w.min()) >= 0.0
    assert torch.allclose(w.sum(-1), torch.ones(w.size(0)), atol=1e-6)


def test_basis_is_continuous_across_a_zone_seam():
    b = RegionalRationalBasis(2, 5, zones=4, bases_per_zone=1, degrees=(8, 6))
    seam = float(b.zone_centre[0] + b.zone_half[0])
    eps = 1e-6
    left = torch.tensor([[0.5 + seam - eps, 0.5]])
    right = torch.tensor([[0.5 + seam + eps, 0.5]])
    with torch.no_grad():
        assert float((b(left) - b(right)).abs().max()) < 1e-4


def test_zone_count_does_not_grow_with_dimension():
    """The reason for radial zones rather than a grid: a grid of k splits per
    axis gives k**dim cells, radial zones give exactly `zones` in any dim."""
    for dim in (2, 3):
        b = RegionalRationalBasis(dim, 5, zones=6, bases_per_zone=1,
                                  degrees=(8, 6))
        assert b.num_bases == 6
        assert b(grid(21, dim=dim)).shape[-1] == 6


def test_default_radii_are_the_measured_spair_quantiles():
    assert DEFAULT_ZONE_RADII[(2, 4)][0] == 0.0
    assert DEFAULT_ZONE_RADII[(2, 4)][-1] == pytest.approx(2 ** 0.5 / 2, abs=1e-3)
    b = RegionalRationalBasis(2, 5, zones=4, bases_per_zone=1, degrees=(8, 6))
    assert b.zone_centre.numel() == 4 and bool((b.zone_half > 0).all())


def test_bad_settings_are_rejected():
    with pytest.raises(AssertionError, match="init='zone'"):
        RegionalRationalBasis(2, 5, zones=4, init='pca')
    with pytest.raises(AssertionError, match='num_bases'):
        RegionalRationalBasis(2, 5, zones=4, bases_per_zone=1, num_bases=7)
    with pytest.raises(AssertionError, match='zone_overlap'):
        RegionalRationalBasis(2, 5, zones=4, zone_overlap=1.0)
    with pytest.raises(AssertionError, match='increase'):
        RegionalRationalBasis(2, 5, zones=2, zone_radii=[0.0, 0.5, 0.2])


def test_conv_forward_backward_and_vp():
    conv = RationalConv(16, 8, dim=2, kernel_size=5, basis='regional',
                        zones=4, bases_per_zone=1, degrees=(8, 6), vp=True)
    assert conv.num_bases == 4 and conv.weight.shape == (4, 16, 8)
    assert conv.gain > 0 and torch.isfinite(torch.tensor(conv.gain))
    x = torch.randn(12, 16)
    edge_index = torch.randint(0, 12, (2, 30))
    out = conv(x, edge_index, torch.rand(30, 2))
    assert out.shape == (12, 8) and torch.isfinite(out).all()
    out.sum().backward()
    grads = [p.grad for p in conv.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_regional_matches_its_comparators_in_parameter_count():
    """zone_K4 must cost exactly what mv_K4 costs, or the comparison is not
    matched. Builds the DGMC stack the SPair experiment builds."""
    def total(flags):
        a = args_for(flags)
        psi_1 = make_backbone(a, 1024, 256, 2, 2, cat=False, dropout=0.5)
        psi_2 = make_backbone(a, 128, 128, 2, 2, cat=True, dropout=0.0)
        return sum(p.numel() for p in DGMC(psi_1, psi_2, num_steps=10).parameters())

    zone4 = total('--backbone rational --rational_basis regional --zones 4 '
                  '--bases_per_zone 1 --degrees 8 6 --vp')
    mv4 = total('--backbone rational --rational_basis multivariate '
                '--degrees 8 6 --init pca --vp --num_bases 4')
    assert zone4 == mv4
    zone9 = total('--backbone rational --rational_basis regional --zones 3 '
                  '--bases_per_zone 3 --degrees 8 6 --vp')
    mv9 = total('--backbone rational --rational_basis multivariate '
                '--degrees 8 6 --init pca --vp --num_bases 9')
    assert zone9 == mv9


def test_freeze_basis_freezes_only_the_shapes():
    flags = ('--backbone rational --rational_basis multivariate --degrees 8 6 '
             '--init pca --vp --num_bases 4')
    a = args_for(flags)
    model = make_backbone(a, 64, 32, 2, 2, cat=False, dropout=0.0)
    assert freeze_basis(model, a) == 0            # flag not set

    a = args_for(flags + ' --freeze_basis')
    model = make_backbone(a, 64, 32, 2, 2, cat=False, dropout=0.0)
    frozen = freeze_basis(model, a)
    assert frozen > 0
    for name, p in model.named_parameters():
        assert p.requires_grad != name.endswith(('numerator', 'denominator'))

    a.lr = 1e-3
    opt = make_optimizer(model, a)
    seen = sum(p.numel() for g in opt.param_groups for p in g['params'])
    assert seen == sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_regional_basis_never_takes_the_fused_path():
    r"""The fused Triton kernels receive only the coefficient tensors and a
    spec tuple, so they cannot see `windows`; the parent's `forward` returns
    their result without calling `evaluate`, where the window lives. If that
    path were ever taken the zone windows would vanish with no error and
    `zone_*` runs would silently become plain multivariate runs, so the guard
    that keeps this basis eager is asserted directly."""
    basis = RegionalRationalBasis(2, 5, zones=3, bases_per_zone=3,
                                  degrees=(8, 6))
    # Shapes and dtypes that satisfy triton_basis.available() on CUDA.
    u = torch.rand(64, 2, dtype=torch.float32)
    assert basis.fused(u) is False
    assert torch.equal(basis(u), basis.evaluate(u, basis.numerator,
                                                basis.denominator))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_regional_forward_matches_evaluate_on_cuda():
    r"""The regression this guards against only fires on CUDA, so assert the
    invariant there too: `forward` must equal the windowed `evaluate`."""
    basis = RegionalRationalBasis(2, 5, zones=3, bases_per_zone=3,
                                  degrees=(8, 6)).cuda()
    u = torch.rand(256, 2, dtype=torch.float32, device='cuda')
    out, ref = basis(u), basis.evaluate(u, basis.numerator, basis.denominator)
    assert torch.equal(out, ref)
    # A window that had been dropped would show up as a much larger row sum:
    # the Shepard-normalised windows are a partition of unity over the zones.
    assert out.abs().sum(-1).max() < ref.abs().sum(-1).max() * 1.001
