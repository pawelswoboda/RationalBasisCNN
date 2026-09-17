r"""Fused Triton kernels for the multivariate safe-Padé rational basis of
:class:`~rational_cnn.rational.MultivariateRationalBasis`,

.. math::
    R_p(\mathbf{u}) = \frac{P_p(\mathbf{t})}{Q_p(\mathbf{t})}, \quad
    P_p = \sum_{\alpha} a_{p\alpha} \phi_\alpha(\mathbf{t}), \quad
    Q_p = 1 + \Big|\sum_{\alpha} b_{p\alpha} \phi_\alpha(\mathbf{t})\Big|
    \;\text{(safe B)}\;\; \text{or}\;\;
    1 + \sum_\alpha |b_{p\alpha}| |\phi_\alpha(\mathbf{t})| \;\text{(safe A)},

with :math:`\mathbf{t} = 2\,\mathrm{clamp}(\mathbf{u}, 0, 1) - 1` and
:math:`\phi_\alpha(\mathbf{t}) = \prod_d T_{\alpha_d}(t_d)` Chebyshev (or
monomial) products over all multi-indices of bounded total degree.

Each program handles a block of edges and evaluates all :math:`K` basis
functions in registers: the one-dimensional polynomial recurrences and the
sum over the (typically 100-250) multi-indices are unrolled at code-generation
time for the given :obj:`(D, degrees, K, safe, poly)`. Per edge the forward
kernel reads :math:`D` coordinates and writes :math:`K` values; no feature
tensor is ever materialised, so memory is :math:`O(E K)` instead of
:math:`O(E F)` and the pass is bandwidth-bound on the output alone. The
backward kernel (gradients w.r.t. the coefficients only; pseudo-coordinates
carry no gradient) recomputes the forward quantities and reduces the
per-block partial sums :math:`\sum_e g_{ep} \partial R_p / \partial c` in
registers, writing one partial vector per block that is summed afterwards
(deterministic, no atomics).

Numerically the polynomials are evaluated as nested sums over the
multi-index (Horner-like, dimension 0 outermost) with the one-dimensional
recurrences generated inside the loop that consumes them, which keeps ~12
vectors live per lane (all :math:`D(m+1)` features live at once would spill
registers). The backward reduction processes groups of ``G`` terms (default
64) per kernel over ``ITERS`` sub-blocks; the grid puts the :math:`K` basis
functions of an edge block adjacent so that the coordinates are read from L2.

Measured on an RTX 4070 Ti for 12.4M edges, :math:`D = 3`, degrees (8, 6),
:math:`K = 8` (forward + backward): eager 610 ms, ``torch.compile`` 210 ms,
these kernels ~40 ms, at 3 GB peak memory and without chunking or
checkpointing.
"""
import math
import os
from functools import lru_cache

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False


# Implementation used when RATIONAL_BASIS_IMPL is not set: 'triton' (these
# kernels), 'compile' (chunked torch.compile'd evaluation) or 'eager'.
DEFAULT_IMPL = 'triton'


def available(u):
    r"""Whether the fused kernels can evaluate the basis at :obj:`u`."""
    impl = os.environ.get('RATIONAL_BASIS_IMPL', DEFAULT_IMPL)
    return HAS_TRITON and impl == 'triton' and u.is_cuda and u.dim() == 2 \
        and u.dtype == torch.float32 and not u.requires_grad


def _nested_poly_code(w, D, indices, coef_ptr, acc, degree, poly, ind, abs_phi=False,
                      abs_coef=False):
    r"""Emits ``acc += coef[f] * prod_d T_{alpha_d}(t_d)`` over all
    multi-indices in :obj:`indices` as *nested* sums (Horner-like): the loop
    over dimension 0 is outermost, and the one-dimensional polynomial values
    are produced by their recurrence inside the loop that needs them, so only
    a handful of vectors are live at any time (instead of all
    :math:`D (m + 1)` features)."""
    lookup = {alpha: f for f, alpha in enumerate(indices)}

    def rec(level, prefix, remaining, target):
        # target: accumulator name for sum over this level's index.
        t = f't{level}'
        for j in range(remaining + 1):
            alpha_j = prefix + (j,)
            # recurrence for T_j(t_level), variables named per (level, path)
            name = f'{ind}T{level}_{j}'
            if j == 0:
                w(f'    {name} = {t} * 0.0 + 1.0')
            elif j == 1:
                w(f'    {name} = {t}')
            elif poly == 'chebyshev':
                w(f'    {name} = 2.0 * {t} * {ind}T{level}_{j - 1} - {ind}T{level}_{j - 2}')
            else:
                w(f'    {name} = {t} * {ind}T{level}_{j - 1}')
            if level == D - 1:
                f = lookup.get(alpha_j)
                if f is None:
                    continue
                c = f'tl.load({coef_ptr} + {f})'
                if abs_coef:
                    c = f'tl.abs({c})'
                phi = f'tl.abs({name})' if abs_phi else name
                w(f'    {target} += {c} * {phi}')
            else:
                inner = f'{ind}acc{level + 1}'
                w(f'    {inner} = {t} * 0.0')
                rec(level + 1, alpha_j, remaining - j, inner)
                phi = f'tl.abs({name})' if abs_phi else name
                w(f'    {target} += {phi} * {inner}')

    rec(0, (), degree, acc)


def _generate(D, num_indices, den_indices, K, safe, poly, G=16):
    r"""Kernel source. Grid: (edge blocks, K); program (pid, p) handles basis
    function p for BLOCK edges."""
    Fn, Fd = len(num_indices), len(den_indices)
    m = max(sum(a) for a in num_indices)
    n = max(sum(a) for a in den_indices)
    L = []
    w = L.append
    w('import triton')
    w('import triton.language as tl')
    w('')

    def load_coords():
        for d in range(D):
            w(f'    t{d} = tl.load(u_ptr + offs * {D} + {d}, mask=mask, other=0.0)')
            w(f'    t{d} = 2.0 * tl.minimum(tl.maximum(t{d}, 0.0), 1.0) - 1.0')

    def evaluate_PQ():
        w(f'    ap = a_ptr + p * {Fn}')
        w(f'    bp = b_ptr + p * {Fd}')
        w('    P = t0 * 0.0')
        _nested_poly_code(w, D, num_indices, 'ap', 'P', m, poly, 'n')
        w('    S = t0 * 0.0')
        _nested_poly_code(w, D, den_indices, 'bp', 'S', n, poly, 'd',
                          abs_phi=(safe == 'A'), abs_coef=(safe == 'A'))
        w('    Q = 1.0 + tl.abs(S)' if safe == 'B' else '    Q = 1.0 + S')

    # ------------------------------------------------------------ forward
    # out is stored basis-major [K, E] (coalesced); transposed by the caller.
    w('@triton.jit')
    w('def fwd(u_ptr, a_ptr, b_ptr, out_ptr, E, BLOCK: tl.constexpr):')
    w('    p = tl.program_id(0)')
    w('    pid = tl.program_id(1)')
    w('    offs = pid * BLOCK + tl.arange(0, BLOCK)')
    w('    mask = offs < E')
    load_coords()
    evaluate_PQ()
    w('    tl.store(out_ptr + p * E + offs, P / Q, mask=mask)')
    w('')
    # ------------------------------------------------------------ backward
    # (1) per-edge chain factors cA = g / Q, cB = -g P / Q^2 sign(S), both
    # stored basis-major [K, E].
    w('@triton.jit')
    w('def bwd_coef(u_ptr, a_ptr, b_ptr, g_ptr, cA_ptr, cB_ptr, E, BLOCK: tl.constexpr):')
    w('    p = tl.program_id(0)')
    w('    pid = tl.program_id(1)')
    w('    offs = pid * BLOCK + tl.arange(0, BLOCK)')
    w('    mask = offs < E')
    load_coords()
    evaluate_PQ()
    w(f'    g = tl.load(g_ptr + offs * {K} + p, mask=mask, other=0.0)')
    w('    tl.store(cA_ptr + p * E + offs, g / Q, mask=mask)')
    if safe == 'B':
        w('    sg = tl.where(S > 0, 1.0, tl.where(S < 0, -1.0, 0.0))')
        w('    tl.store(cB_ptr + p * E + offs, -g * P / (Q * Q) * sg, mask=mask)')
    else:
        w('    tl.store(cB_ptr + p * E + offs, -g * P / (Q * Q), mask=mask)')
    w('')
    # (2) reduction over edges: one kernel per group of <= G terms; program
    # (pid, p) accumulates the group's terms per lane over ITERS sub-blocks
    # of BLOCK edges and reduces each term once at the end. Terms are grouped
    # in nested order so that few one-dimensional features are needed.
    terms = [(f, alpha, 'A') for f, alpha in enumerate(num_indices)] + \
            [(Fn + f, alpha, 'B') for f, alpha in enumerate(den_indices)]
    groups = []
    for kind in ['A', 'B']:
        ts = sorted([t for t in terms if t[2] == kind], key=lambda t: t[1])
        groups += [ts[i:i + G] for i in range(0, len(ts), G)]
    F = Fn + Fd
    for gi, grp in enumerate(groups):
        needed = {d: sorted({alpha[d] for _, alpha, _ in grp}) for d in range(D)}
        w('@triton.jit')
        w(f'def bwd_red_{gi}(u_ptr, c_ptr, part_ptr, E, ITERS, BLOCK: tl.constexpr):')
        w('    p = tl.program_id(0)')
        w('    pid = tl.program_id(1)')
        w('    lane = tl.arange(0, BLOCK)')
        for j in range(len(grp)):
            w(f'    acc{j} = tl.zeros([BLOCK], dtype=tl.float32)')
        w('    for it in range(ITERS):')
        w('        offs = (pid * ITERS + it) * BLOCK + lane')
        w('        mask = offs < E')
        for d in range(D):
            w(f'        t{d} = tl.load(u_ptr + offs * {D} + {d}, mask=mask, other=0.0)')
            w(f'        t{d} = 2.0 * tl.minimum(tl.maximum(t{d}, 0.0), 1.0) - 1.0')
        for d in range(D):  # features up to the max degree needed in dim d
            top = max(needed[d])
            for j in range(top + 1):
                if j == 0:
                    w(f'        T{d}_0 = t{d} * 0.0 + 1.0')
                elif j == 1:
                    w(f'        T{d}_1 = t{d}')
                elif poly == 'chebyshev':
                    w(f'        T{d}_{j} = 2.0 * t{d} * T{d}_{j - 1} - T{d}_{j - 2}')
                else:
                    w(f'        T{d}_{j} = t{d} * T{d}_{j - 1}')
        w(f'        c = tl.load(c_ptr + p * E + offs, mask=mask, other=0.0)')
        for j, (f, alpha, kind) in enumerate(grp):
            factors = [f'T{d}_{a}' for d, a in enumerate(alpha) if a > 0]
            phi = ' * '.join(factors) if factors else 'T0_0'
            if kind == 'B' and safe == 'A':
                phi = f'tl.abs({phi})'
            w(f'        acc{j} += c * ({phi})')
        for j, (f, alpha, kind) in enumerate(grp):
            w(f'    tl.store(part_ptr + (pid * {K} + p) * {F} + {f}, tl.sum(acc{j}, axis=0))')
        w('')
    w('GROUP_KINDS = ' + repr([grp[0][2] for grp in groups]))
    w('bwd_red = [' + ', '.join(f'bwd_red_{gi}' for gi in range(len(groups))) + ']')
    return '\n'.join(L) + '\n'


class _Kernels:
    r"""Generated kernels are written to a cache directory
    (``$RATIONAL_KERNEL_CACHE`` or ``~/.cache/rational_cnn/triton``) and
    imported from there, since Triton needs the source on disk."""
    def __init__(self, src):
        import hashlib
        import importlib.util
        self.src = src
        cache = os.environ.get('RATIONAL_KERNEL_CACHE',
                               os.path.expanduser('~/.cache/rational_cnn/triton'))
        os.makedirs(cache, exist_ok=True)
        name = 'rational_basis_' + hashlib.sha1(src.encode()).hexdigest()[:16]
        path = os.path.join(cache, name + '.py')
        if not os.path.exists(path):
            tmp = path + f'.{os.getpid()}.tmp'
            with open(tmp, 'w') as f:
                f.write(src)
            os.replace(tmp, path)
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.fwd, self.bwd_coef = mod.fwd, mod.bwd_coef
        self.bwd_red, self.group_kinds = mod.bwd_red, mod.GROUP_KINDS


@lru_cache(maxsize=None)
def get_kernels(D, num_indices, den_indices, K, safe, poly):
    G = int(os.environ.get('RATIONAL_TRITON_GROUP', 64))
    return _Kernels(_generate(D, list(num_indices), list(den_indices), K,
                              safe, poly, G=G))


# Launch configuration: BLOCK edges per program (one basis function each).
FWD_BLOCK = int(os.environ.get('RATIONAL_TRITON_BLOCK', 512))
FWD_WARPS = int(os.environ.get('RATIONAL_TRITON_WARPS', 4))
# Backward reduction: programs of ITERS sub-blocks of RED_BLOCK edges.
RED_BLOCK = int(os.environ.get('RATIONAL_TRITON_RED_BLOCK', 256))
RED_WARPS = int(os.environ.get('RATIONAL_TRITON_RED_WARPS', 4))
RED_ITERS = int(os.environ.get('RATIONAL_TRITON_RED_ITERS', 16))


class _RationalBasisFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, a, b, spec):
        D, num_indices, den_indices, safe, poly = spec
        K = a.size(0)
        kern = get_kernels(D, num_indices, den_indices, K, safe, poly)
        u, a, b = u.contiguous(), a.contiguous().float(), b.contiguous().float()
        E = u.size(0)
        out = torch.empty(K, E, device=u.device, dtype=torch.float32)
        if E > 0:
            grid = (K, triton.cdiv(E, FWD_BLOCK))
            kern.fwd[grid](u, a, b, out, E, BLOCK=FWD_BLOCK, num_warps=FWD_WARPS)
        ctx.save_for_backward(u, a, b)
        ctx.spec, ctx.kern = spec, kern
        return out.t().contiguous()

    @staticmethod
    def backward(ctx, g):
        u, a, b = ctx.saved_tensors
        safe = ctx.spec[3]
        K, Fn = a.shape
        Fd = b.size(1)
        E = u.size(0)
        kern = ctx.kern
        if E == 0:
            return None, torch.zeros_like(a), torch.zeros_like(b), None
        g = g.contiguous().float()
        cA = torch.empty(K, E, device=u.device, dtype=torch.float32)
        cB = torch.empty_like(cA)
        kern.bwd_coef[(K, triton.cdiv(E, FWD_BLOCK))](
            u, a, b, g, cA, cB, E, BLOCK=FWD_BLOCK, num_warps=FWD_WARPS)
        n_prog = triton.cdiv(E, RED_BLOCK * RED_ITERS)
        part = torch.empty(n_prog, K, Fn + Fd, device=u.device,
                           dtype=torch.float32)
        for red, kind in zip(kern.bwd_red, kern.group_kinds):
            red[(K, n_prog)](u, cA if kind == 'A' else cB, part, E, RED_ITERS,
                             BLOCK=RED_BLOCK, num_warps=RED_WARPS)
        grad = part.sum(0)
        ga, gb = grad[:, :Fn], grad[:, Fn:]
        if safe == 'A':
            gb = gb * torch.sign(b)
        return None, ga.to(a.dtype), gb.contiguous().to(b.dtype), None


def rational_basis(u, numerator, denominator, num_indices, den_indices,
                   safe='B', poly='chebyshev'):
    r"""Evaluates the multivariate rational basis at :obj:`u` of shape
    :obj:`[E, D]` (CUDA, float32) with the fused kernels; returns
    :obj:`[E, K]`. Differentiable w.r.t. :obj:`numerator` / :obj:`denominator`
    (:obj:`[K, F_num]` / :obj:`[K, F_den]`)."""
    spec = (u.size(1), tuple(map(tuple, num_indices)),
            tuple(map(tuple, den_indices)), safe, poly)
    return _RationalBasisFn.apply(u, numerator, denominator, spec)
