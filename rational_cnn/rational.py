import math
from itertools import product

import torch
from torch.nn import Linear as Lin, Parameter
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import uniform

from .bspline import open_bspline_basis_1d, bspline_basis
from .large import basis_conv, basis_conv_max, chunked_basis


def poly_features(t, degree, poly='chebyshev'):
    r"""Polynomial features :math:`[\phi_0(t), \dots, \phi_{\mathrm{degree}}
    (t)]` of shape :obj:`[..., degree + 1]` for :math:`t \in [-1, 1]`."""
    feats = [torch.ones_like(t)]
    if degree >= 1:
        feats.append(t)
    for j in range(2, degree + 1):
        if poly == 'monomial':
            feats.append(feats[-1] * t)
        elif poly == 'chebyshev':
            feats.append(2 * t * feats[-1] - feats[-2])
        else:
            raise ValueError("Unknown polynomial basis '{}'".format(poly))
    return torch.stack(feats, dim=-1)


def _refine_fit(evaluate, u, target, a, b, fit_steps):
    r"""Refines a least-squares numerator fit :obj:`a` (with small random
    denominator :obj:`b`) of :obj:`evaluate(u, a, b)` to :obj:`target` by
    :obj:`fit_steps` Adam steps followed by an L-BFGS polish."""
    with torch.enable_grad():
        a, b = a.requires_grad_(), b.requires_grad_()

        def loss_fn():
            return F.mse_loss(evaluate(u, a, b), target)

        opt = torch.optim.Adam([a, b], lr=0.02)
        for _ in range(fit_steps):
            opt.zero_grad()
            loss_fn().backward()
            opt.step()

        opt = torch.optim.LBFGS([a, b], max_iter=100,
                                line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss

        opt.step(closure)
    return a.detach(), b.detach()


def total_degree_indices(dim, degree):
    r"""All multi-indices :math:`\alpha \in \mathbb{N}^{\mathrm{dim}}` with
    :math:`|\alpha| \le \mathrm{degree}`, ordered by total degree (the
    constant term first)."""
    idx = [a for a in product(range(degree + 1), repeat=dim)
           if sum(a) <= degree]
    return sorted(idx, key=lambda a: (sum(a), a[::-1]))


def multivariate_poly_features(t, indices, poly='chebyshev'):
    r"""Products :math:`\prod_d \phi_{\alpha_d}(t_d)` of one-dimensional
    polynomial features for every multi-index :math:`\alpha` in
    :obj:`indices`, for :obj:`t` of shape :obj:`[..., dim]`; returns a
    tensor of shape :obj:`[..., len(indices)]`."""
    dim = t.size(-1)
    degree = max(max(a) for a in indices)
    # [..., dim, degree + 1] one-dimensional features, then one gather per
    # dimension (vectorised over the multi-indices).
    phi = torch.stack([poly_features(t[..., d], degree, poly)
                       for d in range(dim)], dim=-2)
    idx = torch.tensor(indices, device=t.device, dtype=torch.long)
    out = phi[..., 0, :].index_select(-1, idx[:, 0])
    for d in range(1, dim):
        out = out * phi[..., d, :].index_select(-1, idx[:, d])
    return out


def gauss_basis_1d(u, kernel_size, width=0.5):
    r"""Gaussian bumps centred at the :obj:`kernel_size` uniform knots on
    :math:`[0, 1]` with :math:`\sigma = \mathrm{width} / (\mathrm{kernel\_size}
    - 1)`, of shape :obj:`[..., kernel_size]` (a smooth RBF analogue of the
    hat basis)."""
    centers = torch.linspace(0, 1, kernel_size, dtype=u.dtype, device=u.device)
    sigma = width / (kernel_size - 1)
    return torch.exp(-0.5 * ((u.unsqueeze(-1) - centers) / sigma) ** 2)


def basis_energy(basis_fn, dim, num_points=None):
    r"""Numerically integrates :math:`\mathbb{E}_{u \sim U[0,1]^D} \sum_p
    B_p(u)^2` for a basis :obj:`basis_fn: [P, D] -> [P, K]` (the quantity
    that sets the output variance of a continuous-kernel convolution)."""
    if num_points is None:  # ~4k-260k grid points regardless of dim
        num_points = {1: 4096, 2: 64, 3: 24}.get(dim, 12)
    axes = [torch.linspace(0, 1, num_points, dtype=torch.double)] * dim
    u = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1).view(-1, dim)
    with torch.no_grad():
        return basis_fn(u).pow(2).sum(dim=-1).mean().item()


def hat_energy(kernel_size):
    r"""`basis_energy` of the tensor-product degree-1 B-spline basis."""
    kernel_size = list(kernel_size)

    def fn(u):
        idx, weight = bspline_basis(u, kernel_size)
        K = 1
        for k in kernel_size:
            K *= k
        return u.new_zeros(u.size(0), K).scatter_add_(1, idx, weight)

    return basis_energy(fn, len(kernel_size))


_FIT_CACHE = {}  # see MultivariateRationalBasis.fit_to_spline


class RationalBasis1D(torch.nn.Module):
    r"""A set of :obj:`num_bases` learnable rational basis functions
    :math:`r_k : [0, 1] \to \mathbb{R}`, parametrized as *safe* Padé
    approximants (`Molina et al., "Padé Activation Units", ICLR 2020
    <https://arxiv.org/abs/1907.06732>`_), *i.e.* with :math:`t = 2u - 1`,

    .. math::
        r_k(u) = \frac{\sum_{j=0}^{m} a_{k,j} \, \phi_j(t)}
                      {1 + \big| \sum_{j=1}^{n} b_{k,j} \, \phi_j(t) \big|}
        \quad \textrm{(version B)}, \qquad
        r_k(u) = \frac{\sum_{j=0}^{m} a_{k,j} \, \phi_j(t)}
                      {1 + \sum_{j=1}^{n} \big| b_{k,j} \, \phi_j(t) \big|}
        \quad \textrm{(version A)},

    where :math:`\phi_j` are monomials or Chebyshev polynomials (the rational
    KAN of `Aghaei, 2024 <https://arxiv.org/abs/2406.14495>`_ uses Jacobi
    polynomials, of which Chebyshev is the best-conditioned special case; it
    uses an unguarded denominator, which is what the safe form fixes).
    The denominator is bounded below by :math:`1`, so the basis can never
    blow up, and every basis function is smooth on :math:`[0, 1]`.

    Args:
        num_bases (int): Number of basis functions :math:`K`.
        degrees ((int, int), optional): Numerator and denominator degrees
            :math:`(m, n)`. (default: :obj:`(5, 4)`)
        safe (str, optional): Safe denominator variant, :obj:`'A'` or
            :obj:`'B'`. (default: :obj:`'B'`)
        poly (str, optional): Polynomial features, :obj:`'chebyshev'` or
            :obj:`'monomial'`. (default: :obj:`'chebyshev'`)
        init (str, optional): :obj:`'spline'` fits each rational function to
            the corresponding degree-1 B-spline (hat) basis function over
            :obj:`num_bases` knots, so that the layer initially behaves like a
            :class:`SplineConv`; :obj:`'random'` starts from a near-constant
            basis :math:`r_k \approx 1/K` with :math:`\mathcal{N}(0, 0.1^2 /
            (m+1))` numerator and :math:`\mathcal{N}(0, 0.01^2)` denominator
            noise; :obj:`'constant'` starts from :math:`r_k \equiv 1` (Padé
            with zero denominator) plus :math:`\mathcal{N}(0,
            \mathrm{init\_noise}^2)` noise on all coefficients.
            (default: :obj:`'spline'`)
        fit_steps (int, optional): Number of optimization steps for the
            :obj:`'spline'` fit after the closed-form polynomial fit.
            (default: :obj:`500`)
        init_noise (float, optional): Noise scale of the :obj:`'constant'`
            init. Must be non-zero: at an exactly zero denominator the
            gradient of :math:`|Q|` vanishes identically and the denominator
            would never be trained. (default: :obj:`1e-3`)
    """
    def __init__(self, num_bases, degrees=(5, 4), safe='B', poly='chebyshev',
                 init='spline', fit_steps=500, init_noise=1e-3):
        super(RationalBasis1D, self).__init__()

        assert safe in ['A', 'B']
        assert poly in ['chebyshev', 'monomial']
        assert init in ['spline', 'random', 'constant', 'gauss', 'cheb']
        assert init != 'constant' or init_noise > 0
        assert init != 'cheb' or num_bases <= degrees[0] + 1, \
            'cheb init needs numerator degree >= num_bases - 1'
        self.num_bases = num_bases
        self.degrees = tuple(degrees)
        self.safe = safe
        self.poly = poly
        self.init = init
        self.fit_steps = fit_steps
        self.init_noise = init_noise

        m, n = self.degrees
        self.numerator = Parameter(torch.Tensor(num_bases, m + 1))
        self.denominator = Parameter(torch.Tensor(num_bases, n))

        self.reset_parameters()

    def reset_parameters(self):
        K, (m, n) = self.num_bases, self.degrees
        if self.init in ['spline', 'gauss']:
            a, b = self.fit_to_spline()
            self.numerator.data.copy_(a)
            self.denominator.data.copy_(b)
        elif self.init == 'constant':
            self.numerator.data.normal_(0, self.init_noise)
            self.numerator.data[:, 0] += 1.0
            self.denominator.data.normal_(0, self.init_noise)
        elif self.init == 'cheb':
            # r_k = phi_k: the (orthogonal) polynomials themselves, the
            # analogue of the identity init of KAT (Yang & Wang, 2025).
            self.numerator.data.normal_(0, self.init_noise)
            self.numerator.data += torch.eye(K, m + 1).to(self.numerator.dtype)
            self.denominator.data.normal_(0, self.init_noise)
        else:
            self.numerator.data.normal_(0, 0.1 / math.sqrt(m + 1))
            self.numerator.data[:, 0] += 1.0 / K
            # Small but non-zero: the gradient of |Q| vanishes exactly at
            # zero denominator coefficients, which would freeze them.
            self.denominator.data.normal_(0, 0.01)

    def target(self, u):
        r"""The basis the :obj:`'spline'` / :obj:`'gauss'` init is fitted
        to, evaluated at :obj:`u` of shape :obj:`[...]`."""
        if self.init == 'gauss':
            return gauss_basis_1d(u, self.num_bases)
        return open_bspline_basis_1d(u, self.num_bases)

    def evaluate(self, u, numerator, denominator):
        m, n = self.degrees
        t = 2 * u.clamp(0, 1) - 1
        phi = poly_features(t, max(m, n), self.poly)  # [..., max(m,n)+1]
        P = phi[..., None, :m + 1] @ numerator.t()  # [..., 1, K]
        Q = phi[..., None, 1:n + 1] * denominator  # [..., K, n]
        if self.safe == 'B':
            Q = 1 + Q.sum(dim=-1).abs()
        else:
            Q = 1 + Q.abs().sum(dim=-1)
        return P.squeeze(-2) / Q

    def forward(self, u):
        r"""Evaluates all basis functions at :obj:`u` of shape :obj:`[...]`,
        returning a tensor of shape :obj:`[..., num_bases]`."""
        return self.evaluate(u, self.numerator, self.denominator)

    @torch.no_grad()
    def fit_to_spline(self, num_points=256):
        r"""Least-squares fits the rational basis to the degree-1 B-spline
        basis with :obj:`num_bases` knots: a closed-form polynomial fit of
        the numerator (denominator zero), refined by a few hundred steps of
        joint gradient descent on numerator and denominator coefficients."""
        K, (m, n) = self.num_bases, self.degrees
        u = torch.linspace(0, 1, num_points, dtype=torch.double)
        target = self.target(u)  # [P, K]
        phi = poly_features(2 * u - 1, m, self.poly)  # [P, m+1]

        a = torch.linalg.lstsq(phi, target).solution.t().contiguous()  # [K, m+1]
        # Non-zero start: the gradient of |Q| vanishes exactly at Q = 0.
        b = torch.randn(K, n, dtype=torch.double) * 0.01

        if self.fit_steps > 0 and n > 0:
            a, b = _refine_fit(self.evaluate, u, target, a, b, self.fit_steps)

        return a.to(self.numerator.dtype), b.to(self.denominator.dtype)

    def __repr__(self):
        return '{}({}, degrees={}, safe={}, poly={}, init={})'.format(
            self.__class__.__name__, self.num_bases, self.degrees, self.safe,
            self.poly, self.init)


class RationalBasis(torch.nn.Module):
    r"""Tensor product of per-dimension :class:`RationalBasis1D` sets,
    mapping pseudo-coordinates of shape :obj:`[E, dim]` to a dense basis of
    shape :obj:`[E, prod(kernel_size)]` (the rational analogue of the
    B-spline basis used in :class:`SplineConv`)."""
    def __init__(self, dim, kernel_size, **kwargs):
        super(RationalBasis, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        assert len(kernel_size) == dim
        self.dim = dim
        self.kernel_size = list(kernel_size)
        self.bases = torch.nn.ModuleList(
            [RationalBasis1D(k, **kwargs) for k in self.kernel_size])

    @property
    def num_bases(self):
        K = 1
        for k in self.kernel_size:
            K *= k
        return K

    def reset_parameters(self):
        for basis in self.bases:
            basis.reset_parameters()

    def forward(self, pseudo):
        E = pseudo.size(0)
        out = pseudo.new_ones(E, 1)
        for d, basis in enumerate(self.bases):
            # Same index ordering (first dimension fastest) as
            # `bspline_basis`, so weights are transferable to `BSplineConv`.
            out = (basis(pseudo[:, d]).unsqueeze(-1) * out.unsqueeze(-2))
            out = out.view(E, -1)
        return out

    def __repr__(self):
        return '{}(dim={}, kernel_size={}, {})'.format(
            self.__class__.__name__, self.dim, self.kernel_size,
            self.bases[0])


class MultivariateRationalBasis(torch.nn.Module):
    r"""A set of :math:`K = \prod_d k_d` learnable *multivariate* rational
    basis functions :math:`B_p : [0, 1]^D \to \mathbb{R}`,

    .. math::
        B_p(\mathbf{u}) = \frac{P_p(\mathbf{t})}{1 + |Q_p(\mathbf{t})|},
        \qquad
        P_p(\mathbf{t}) = \sum_{|\alpha| \le m} a_{p,\alpha}
        \prod_d \phi_{\alpha_d}(t_d), \qquad
        Q_p(\mathbf{t}) = \sum_{1 \le |\alpha| \le n} b_{p,\alpha}
        \prod_d \phi_{\alpha_d}(t_d),

    with :math:`\mathbf{t} = 2\mathbf{u} - 1` and multivariate polynomials of
    *total* degree :math:`m` (numerator) and :math:`n` (denominator) in the
    monomial or Chebyshev product basis. In contrast to
    :class:`RationalBasis`, which multiplies :math:`D` one-dimensional
    rational functions, the numerator and denominator here are single
    polynomials in all :math:`D` pseudo-coordinates, so the basis functions
    need not be axis-aligned (separable). The number of coefficients per
    basis function is :math:`\binom{m + D}{D}` for the numerator and
    :math:`\binom{n + D}{D} - 1` for the denominator.

    Args:
        dim (int): Pseudo-coordinate dimensionality :math:`D`.
        kernel_size (int or [int]): The basis has :math:`\prod_d k_d`
            functions, indexed like the tensor-product B-spline basis of
            :class:`BSplineConv`, so that weights are transferable.
        degrees ((int, int), optional): Total numerator and denominator
            degrees :math:`(m, n)`. (default: :obj:`(5, 4)`)
        safe (str, optional): Safe denominator variant, :obj:`'A'`
            (:math:`1 + \sum |b_\alpha \phi_\alpha|`) or :obj:`'B'`
            (:math:`1 + |\sum b_\alpha \phi_\alpha|`). (default: :obj:`'B'`)
        poly (str, optional): :obj:`'chebyshev'` or :obj:`'monomial'`.
            (default: :obj:`'chebyshev'`)
        init (str, optional): :obj:`'spline'` fits each :math:`B_p` to the
            corresponding tensor-product degree-1 B-spline basis function on
            a grid over :math:`[0, 1]^D`; :obj:`'random'` starts from a
            near-constant basis :math:`B_p \approx 1/K`; :obj:`'constant'`
            from :math:`B_p \equiv 1` plus :math:`\mathcal{N}(0,
            \mathrm{init\_noise}^2)` noise (see :class:`RationalBasis1D`).
            (default: :obj:`'spline'`)
        fit_steps (int, optional): Adam steps of the :obj:`'spline'` fit
            after the closed-form numerator fit. (default: :obj:`500`)
        grid_size (int, optional): Fit grid points per dimension.
            (default: :obj:`1024` / :obj:`32` / :obj:`16` for 1-D / 2-D /
            3-D)
        init_noise (float, optional): Noise scale of the :obj:`'constant'`
            init. (default: :obj:`1e-3`)
    """
    def __init__(self, dim, kernel_size, degrees=(5, 4), safe='B',
                 poly='chebyshev', init='spline', fit_steps=500,
                 grid_size=None, init_noise=1e-3, num_bases=None):
        super(MultivariateRationalBasis, self).__init__()

        assert safe in ['A', 'B']
        assert poly in ['chebyshev', 'monomial']
        assert init in ['spline', 'random', 'constant', 'gauss', 'cheb', 'pca']
        assert init != 'constant' or init_noise > 0
        self.init_noise = init_noise
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        assert len(kernel_size) == dim
        self.dim = dim
        self.kernel_size = list(kernel_size)
        self.degrees = tuple(degrees)
        self.safe = safe
        self.poly = poly
        self.init = init
        self.fit_steps = fit_steps
        # Fit grid: ~1k-4k points regardless of dimension.
        self.grid_size = grid_size or {1: 1024, 2: 32, 3: 16}.get(dim, 8)

        # Number of basis functions: the grid size prod(kernel_size) by
        # default (SplineConv-compatible weights), or any K (then the grid
        # only defines the hat basis the 'spline'/'gauss'/'pca' inits refer
        # to; 'spline'/'gauss' need K == prod(kernel_size), 'pca' any K).
        self._num_bases = num_bases
        K = self.num_bases
        if init in ['spline', 'gauss']:
            assert K == self.grid_bases, \
                "init='{}' needs num_bases == prod(kernel_size); use " \
                "init='pca' for a free number of bases".format(init)

        m, n = self.degrees
        self.num_indices = total_degree_indices(dim, m)
        self.den_indices = total_degree_indices(dim, n)[1:]  # no constant
        assert init != 'cheb' or K <= len(self.num_indices), \
            'cheb init needs at least num_bases numerator terms'
        self.numerator = Parameter(torch.Tensor(K, len(self.num_indices)))
        self.denominator = Parameter(torch.Tensor(K, len(self.den_indices)))

        self.reset_parameters()

    @property
    def grid_bases(self):
        K = 1
        for k in self.kernel_size:
            K *= k
        return K

    @property
    def num_bases(self):
        return self._num_bases or self.grid_bases

    def reset_parameters(self):
        K, F_num = self.numerator.size()
        if self.init in ['spline', 'gauss', 'pca']:
            a, b = self.fit_to_spline()
            self.numerator.data.copy_(a)
            self.denominator.data.copy_(b)
        elif self.init == 'constant':
            self.numerator.data.normal_(0, self.init_noise)
            self.numerator.data[:, 0] += 1.0
            self.denominator.data.normal_(0, self.init_noise)
        elif self.init == 'cheb':
            # B_p = the p-th (lowest total degree first) product Chebyshev
            # polynomial.
            self.numerator.data.normal_(0, self.init_noise)
            self.numerator.data += torch.eye(K, F_num).to(self.numerator.dtype)
            self.denominator.data.normal_(0, self.init_noise)
        else:
            self.numerator.data.normal_(0, 0.1 / math.sqrt(F_num))
            self.numerator.data[:, 0] += 1.0 / K
            self.denominator.data.normal_(0, 0.01)

    def target(self, u):
        r"""The tensor-product basis the :obj:`'spline'` / :obj:`'gauss'`
        init is fitted to, for :obj:`u` of shape :obj:`[P, dim]`."""
        if self.init == 'gauss':
            out = u.new_ones(u.size(0), 1)
            for d, k in enumerate(self.kernel_size):
                g = gauss_basis_1d(u[:, d], k)  # [P, k]
                out = (g.unsqueeze(-1) * out.unsqueeze(-2)).view(u.size(0), -1)
            return out
        idx, weight = bspline_basis(u, self.kernel_size)
        hats = u.new_zeros(u.size(0), self.grid_bases).scatter_add_(
            1, idx, weight)
        if self.init == 'pca':
            # The K leading principal directions of the hat basis in
            # function space (left singular vectors of the [P, grid_bases]
            # matrix of hat functions): the K-dimensional subspace that
            # best approximates the SplineConv kernel space. Each
            # component is rescaled to unit maximum, like a hat.
            K = self.num_bases
            assert K <= self.grid_bases, \
                'pca init: num_bases must not exceed prod(kernel_size)'
            U, S, _ = torch.linalg.svd(hats, full_matrices=False)
            comp = U[:, :K]
            # Deterministic sign: largest-magnitude entry positive.
            peak = comp.gather(0, comp.abs().argmax(dim=0, keepdim=True))
            comp = comp * peak.sign()
            return comp / comp.abs().max(dim=0, keepdim=True).values
        return hats

    def features(self, u):
        r"""Numerator and denominator polynomial features of :obj:`u` of
        shape :obj:`[..., dim]`."""
        t = 2 * u.clamp(0, 1) - 1
        indices = self.num_indices + self.den_indices
        phi = multivariate_poly_features(t, indices, self.poly)
        return phi[..., :len(self.num_indices)], phi[..., len(self.num_indices):]

    def evaluate(self, u, numerator, denominator):
        phi_num, phi_den = self.features(u)
        P = phi_num @ numerator.t()  # [..., K]
        if self.safe == 'B':
            Q = 1 + (phi_den @ denominator.t()).abs()
        else:
            Q = 1 + phi_den.abs() @ denominator.abs().t()
        return P / Q

    def fused(self, pseudo):
        r"""Whether :meth:`forward` will use the fused Triton kernels
        (:mod:`rational_cnn.triton_basis`) for :obj:`pseudo`."""
        from . import triton_basis
        return triton_basis.available(pseudo)

    def forward(self, pseudo):
        r"""Evaluates all basis functions at :obj:`pseudo` of shape
        :obj:`[E, dim]`, returning a tensor of shape :obj:`[E, num_bases]`.
        On CUDA the fused Triton kernels are used (set
        :obj:`RATIONAL_BASIS_IMPL=eager` to disable)."""
        if self.fused(pseudo):
            from .triton_basis import rational_basis
            return rational_basis(pseudo, self.numerator, self.denominator,
                                  self.num_indices, self.den_indices,
                                  self.safe, self.poly)
        return self.evaluate(pseudo, self.numerator, self.denominator)

    @property
    def _fit_key(self):
        return ('multivariate', self.dim, tuple(self.kernel_size),
                self.num_bases, self.degrees, self.safe, self.poly,
                self.init, self.fit_steps, self.grid_size)

    @torch.no_grad()
    def fit_to_spline(self):
        r"""Least-squares fits the basis to :meth:`target` (the tensor-product
        degree-1 B-spline basis, Gaussian bumps, or the hat basis' principal
        components) on a :obj:`grid_size**dim` grid: closed-form numerator
        fit, then joint refinement (see :meth:`RationalBasis1D.fit_to_spline`).
        The result is cached per configuration within a process, so stacked
        layers with the same basis share one fit (they differ only in the
        tiny random denominator start)."""
        if self._fit_key in _FIT_CACHE:
            a, b = _FIT_CACHE[self._fit_key]
            return a.clone(), b.clone()
        a, b = self._fit_to_spline()
        _FIT_CACHE[self._fit_key] = (a.clone(), b.clone())
        return a, b

    def _fit_to_spline(self):
        K, (m, n) = self.num_bases, self.degrees
        axes = [torch.linspace(0, 1, self.grid_size, dtype=torch.double)] * self.dim
        u = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)
        u = u.view(-1, self.dim)  # [P, dim]
        target = self.target(u)  # [P, K]
        phi_num, _ = self.features(u)

        a = torch.linalg.lstsq(phi_num, target).solution.t().contiguous()
        b = torch.randn(K, len(self.den_indices), dtype=torch.double) * 0.01

        if self.fit_steps > 0 and n > 0:
            a, b = _refine_fit(self.evaluate, u, target, a, b, self.fit_steps)

        return a.to(self.numerator.dtype), b.to(self.denominator.dtype)

    def __repr__(self):
        return ('{}(dim={}, kernel_size={}, num_bases={}, degrees={}, '
                'safe={}, poly={}, init={})').format(
                    self.__class__.__name__, self.dim, self.kernel_size,
                    self.num_bases, self.degrees, self.safe, self.poly,
                    self.init)


class MLPBasis(torch.nn.Module):
    r"""Control for the rational bases: :obj:`num_bases` kernel basis
    functions given by a two-layer MLP :math:`[0, 1]^D \to \mathbb{R}^K`
    (the filter-generating network of SchNet / PointConv). Accepts and
    ignores the :class:`MultivariateRationalBasis` keyword arguments so it is
    a drop-in replacement inside :class:`RationalConv`.

    Args:
        dim (int): Pseudo-coordinate dimensionality.
        kernel_size (int or [int]): Only used as the default for
            :obj:`num_bases` (:math:`\prod_d k_d`).
        num_bases (int, optional): Number of basis functions :math:`K`.
        hidden (int, optional): Hidden width. (default: :obj:`64`)
    """
    def __init__(self, dim, kernel_size, num_bases=None, hidden=64,
                 **kwargs):
        super(MLPBasis, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        self.dim = dim
        self.kernel_size = list(kernel_size)
        K = 1
        for k in self.kernel_size:
            K *= k
        self._num_bases = num_bases or K
        self.hidden = hidden
        self.lin1 = Lin(dim, hidden)
        self.lin2 = Lin(hidden, self._num_bases)
        self.reset_parameters()

    @property
    def num_bases(self):
        return self._num_bases

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, pseudo):
        t = 2 * pseudo.clamp(0, 1) - 1
        return self.lin2(F.relu(self.lin1(t)))

    def __repr__(self):
        return '{}(dim={}, num_bases={}, hidden={})'.format(
            self.__class__.__name__, self.dim, self.num_bases, self.hidden)


class RationalConv(MessagePassing):
    r"""Continuous-kernel convolution with learnable *rational* (safe Padé)
    basis functions in place of the B-splines of :class:`SplineConv`:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\mathrm{root}} \mathbf{x}_i +
        \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)}
        \Big( \sum_p R_p(\mathbf{u}_{ij}) \mathbf{\Theta}_p \Big) \mathbf{x}_j
        + \mathbf{b},

    with :math:`R_p(\mathbf{u}) = \prod_d r_{p_d}(u_d)` a tensor product of
    learnable one-dimensional rational functions (see
    :class:`RationalBasis1D`). With :obj:`init='spline'`, the layer starts
    as an (approximate) :class:`SplineConv` and learns the shape of its basis
    functions during training.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        dim (int): Pseudo-coordinate dimensionality.
        kernel_size (int or [int]): Number of basis functions per dimension.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not
            learn an additive bias. (default: :obj:`True`)
        basis (str, optional): :obj:`'product'` for a tensor product of
            one-dimensional rational functions (:class:`RationalBasis`),
            :obj:`'multivariate'` for rational functions of multivariate
            polynomials (:class:`MultivariateRationalBasis`), or
            :obj:`'mlp'` for an MLP basis (:class:`MLPBasis`, control).
            (default: :obj:`'product'`)
        aggr (str, optional): Neighbourhood aggregation (:obj:`'mean'`,
            :obj:`'add'`, :obj:`'max'`). (default: :obj:`'mean'`)
        vp (bool, optional): Variance-preserving weight initialisation in
            the spirit of KAT (Yang & Wang, ICLR 2025): the kernel weights
            :math:`\Theta_p` are scaled by :math:`\sqrt{\alpha}` with
            :math:`\alpha = \mathbb{E}_u \sum_p \mathrm{hat}_p(u)^2 /
            \mathbb{E}_u \sum_p B_p(u)^2`, so that the layer produces messages
            of the same variance as a :class:`SplineConv` at initialisation
            whatever the basis init. (default: :obj:`False`)
        **kwargs: Additional arguments of :class:`RationalBasis1D` /
            :class:`MultivariateRationalBasis`.
    """
    def __init__(self, in_channels, out_channels, dim, kernel_size,
                 root_weight=True, bias=True, basis='product', vp=False,
                 aggr='mean', pyg_init=False, large=False, **kwargs):
        super(RationalConv, self).__init__(aggr=aggr, node_dim=0)
        self.pyg_init = pyg_init
        self.large = large

        assert basis in ['product', 'multivariate', 'mlp']
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.vp = vp
        if basis == 'product':
            assert kwargs.pop('num_bases', None) is None, \
                'num_bases is only free for the multivariate / mlp bases'
            self.basis = RationalBasis(dim, kernel_size, **kwargs)
        elif basis == 'multivariate':
            self.basis = MultivariateRationalBasis(dim, kernel_size, **kwargs)
        else:
            self.basis = MLPBasis(dim, kernel_size, **kwargs)
        self.kernel_size = self.basis.kernel_size
        K = self.num_bases = self.basis.num_bases

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
        self.basis.reset_parameters()
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
        self.gain = 1.0
        if self.vp:
            basis = self.basis.double()
            energy = basis_energy(basis, self.dim)
            self.basis.float()
            self.gain = hat_energy(self.kernel_size) / energy
            self.weight.data.mul_(math.sqrt(self.gain))

    def forward(self, x, edge_index, pseudo):
        """"""
        N, K, C_in, C_out = x.size(0), self.num_bases, self.in_channels, \
            self.out_channels
        fused = getattr(self.basis, 'fused', lambda u: False)(pseudo)
        R = chunked_basis(self.basis, pseudo) if self.large and not fused \
            else self.basis(pseudo)  # [E, K]

        if self.large:
            # Basis-first aggregation, O(E C_in) transient memory (see
            # rational_cnn.large), for graphs with ~1e7 edges.
            out = basis_conv_max(R, x, edge_index, self.weight) \
                if self.aggr == 'max' else \
                basis_conv(R, x, edge_index, self.weight, aggr=self.aggr)
        elif C_out <= C_in:
            # Transform node features by all K weight matrices, then combine
            # per edge: memory E * K * C_out.
            xw = x @ self.weight.permute(1, 0, 2).reshape(C_in, -1)
            out = self.propagate(edge_index, xw=xw.view(N, K, C_out), R=R,
                                 x=None)
        else:
            # Kronecker of basis and features, then one matmul: E * K * C_in.
            out = self.propagate(edge_index, x=x, R=R, xw=None)

        if self.root is not None:
            out = out + x @ self.root
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, xw_j, x_j, R):
        if xw_j is not None:
            return torch.einsum('ek,eko->eo', R, xw_j)
        E, K = R.size()
        kron = (R.unsqueeze(-1) * x_j.unsqueeze(-2)).view(E, -1)
        return kron @ self.weight.view(K * self.in_channels, -1)

    def __repr__(self):
        return '{}({}, {}, dim={}, kernel_size={}{}, basis={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.dim, self.kernel_size, ', vp=True' if self.vp else '',
            self.basis)


class RationalCNN(torch.nn.Module):
    r"""Drop-in replacement for :class:`SplineCNN` with :class:`RationalConv`
    layers.

    Args:
        in_channels, out_channels, dim, num_layers, cat, lin, dropout: As in
            :class:`SplineCNN`.
        kernel_size (int, optional): Basis functions per pseudo-coordinate
            dimension. (default: :obj:`5`)
        **kwargs: Additional arguments of :class:`RationalConv`, *e.g.*
            :obj:`basis`, :obj:`degrees`, :obj:`safe`, :obj:`poly`,
            :obj:`init`.
    """
    def __init__(self, in_channels, out_channels, dim, num_layers, cat=True,
                 lin=True, dropout=0.0, kernel_size=5, **kwargs):
        super(RationalCNN, self).__init__()

        self.in_channels = in_channels
        self.dim = dim
        self.num_layers = num_layers
        self.cat = cat
        self.lin = lin
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.basis_kwargs = kwargs

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = RationalConv(in_channels, out_channels, dim, kernel_size,
                                **kwargs)
            self.convs.append(conv)
            in_channels = out_channels

        if self.cat:
            in_channels = self.in_channels + num_layers * out_channels
        else:
            in_channels = out_channels

        if self.lin:
            self.out_channels = out_channels
            self.final = Lin(in_channels, out_channels)
        else:
            self.out_channels = in_channels

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.lin:
            self.final.reset_parameters()

    def forward(self, x, edge_index, edge_attr, *args):
        """"""
        xs = [x]

        for conv in self.convs:
            xs += [F.relu(conv(xs[-1], edge_index, edge_attr))]

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final(x) if self.lin else x
        return x

    def __repr__(self):
        return ('{}({}, {}, dim={}, num_layers={}, cat={}, lin={}, '
                'dropout={}, kernel_size={}, {})').format(
                    self.__class__.__name__, self.in_channels,
                    self.out_channels, self.dim, self.num_layers, self.cat,
                    self.lin, self.dropout, self.kernel_size,
                    self.convs[0].basis)
