r"""Speed / memory of one AEGNN-scale convolution layer: PyG's fused
``torch_spline_conv`` SplineConv vs. our B-spline and rational layers on the
basis-first aggregation of :mod:`rational_cnn.large`, the rational basis
evaluated by the Triton kernels of :mod:`rational_cnn.triton_basis` (and, for
reference, by the ``torch.compile`` / eager chunked evaluation).

Synthetic batch of the N-Caltech101 size: 16 graphs x 25 000 events, ~31
neighbours per node, pseudo-coordinates uniform in [0, 1]^3, kernel size 2
(K = 8 hats). Reports per-layer forward, forward + backward (w.r.t. features
and parameters) and peak memory, then the cost of the basis alone.

    python experiments/bench_kernels.py [--edges 12400000] [--nodes 400000]

Output for an RTX 4070 Ti Super is in ``results/bench_kernels.txt``.
"""
import argparse
import os
import os.path as osp
import sys
import time

import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from rational_cnn.bspline import BSplineConv  # noqa: E402
from rational_cnn.rational import RationalConv  # noqa: E402
from aegnn_net import PygSplineConv  # noqa: E402


def bench(fn, reps=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / reps * 1e3
    peak = (torch.cuda.max_memory_allocated() - base) / 2**30
    return dt, peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=16 * 25000)
    parser.add_argument('--edges', type=int, default=12_400_000)
    parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--kernel_size', type=int, default=2)
    parser.add_argument('--channels', type=int, nargs='+', default=[32, 16, 8],
                        help='in-channels to test (out = in, or 16 for in < 16)')
    parser.add_argument('--reps', type=int, default=10)
    args = parser.parse_args()
    dev = 'cuda'
    torch.manual_seed(0)
    N, E, D, k = args.nodes, args.edges, args.dim, args.kernel_size
    pseudo = torch.rand(E, D, device=dev)
    edge_index = torch.randint(0, N, (2, E), device=dev)
    edge_index = edge_index[:, edge_index[1].argsort()]  # radius_graph order

    def layer_fn(conv, x):
        def f():
            x.grad = None
            conv(x, edge_index, pseudo).sum().backward()
        return f

    def fwd_fn(conv, x):
        def f():
            with torch.no_grad():
                conv(x, edge_index, pseudo)
        return f

    def rational(C_in, C_out, K, degrees):
        return RationalConv(C_in, C_out, D, k, root_weight=False, bias=False,
                            basis='multivariate', aggr='mean', large=True,
                            degrees=degrees, init='pca', num_bases=K)

    print(f'{torch.cuda.get_device_name()}: N={N} E={E} D={D} k={k} '
          f'(E/N={E / N:.1f}), {args.reps} reps')
    print(f'{"layer":58s} {"fwd ms":>8s} {"fwd+bwd ms":>11s} {"peak GB":>8s}')
    for C_in in args.channels:
        C_out = max(C_in, 16)
        x = torch.randn(N, C_in, device=dev, requires_grad=True)
        models = [
            ('PyG SplineConv (torch_spline_conv fused, K=8)',
             PygSplineConv(C_in, C_out, D, k, aggr='mean')),
            ('BSplineConv dense basis + basis-first aggr (K=8)',
             BSplineConv(C_in, C_out, D, k, root_weight=False, bias=False,
                         aggr='mean', large=True)),
            ('RationalConv triton, K=8, deg (8,6)', rational(C_in, C_out, 8, (8, 6))),
            ('RationalConv triton, K=8, deg (5,4)', rational(C_in, C_out, 8, (5, 4))),
            ('RationalConv triton, K=4, deg (8,6)', rational(C_in, C_out, 4, (8, 6))),
        ]
        if C_in == args.channels[0]:
            models += [('RationalConv compile, K=8, deg (8,6)', 'compile'),
                       ('RationalConv eager, K=8, deg (8,6)', 'eager')]
        for name, conv in models:
            env = {}
            if isinstance(conv, str):
                env = {'RATIONAL_BASIS_IMPL': conv}
                if conv == 'eager':
                    env['RATIONAL_NO_COMPILE'] = '1'
                conv = rational(C_in, C_out, 8, (8, 6))
            os.environ.update(env)
            conv = conv.to(dev)
            reps = 3 if env else args.reps
            try:
                f_ms, _ = bench(fwd_fn(conv, x), reps=reps)
                fb_ms, peak = bench(layer_fn(conv, x), reps=reps)
                print(f'C {C_in:2d}->{C_out:2d} {name:50s} {f_ms:8.1f} '
                      f'{fb_ms:11.1f} {peak:8.2f}')
            except torch.cuda.OutOfMemoryError:
                print(f'C {C_in:2d}->{C_out:2d} {name:50s}      OOM')
            for key in env:
                os.environ.pop(key, None)
            del conv
            torch.cuda.empty_cache()

    print('\nbasis only (no aggregation):')
    from torch_spline_conv import spline_basis
    ks = torch.tensor([k] * D, device=dev)
    is_open = torch.ones(D, dtype=torch.uint8, device=dev)
    ms, peak = bench(lambda: spline_basis(pseudo, ks, is_open, 1))
    print(f'{"torch_spline_conv spline_basis fwd (K=8)":58s} {ms:8.1f} ms '
          f'{peak:6.2f} GB')
    basis = rational(32, 32, 8, (8, 6)).to(dev).basis

    def rb_f():
        with torch.no_grad():
            basis(pseudo)

    ms, peak = bench(rb_f)
    print(f'{"triton rational basis fwd (K=8, deg 8,6)":58s} {ms:8.1f} ms '
          f'{peak:6.2f} GB')
    ms, peak = bench(lambda: basis(pseudo).sum().backward())
    print(f'{"triton rational basis fwd+bwd (K=8, deg 8,6)":58s} {ms:8.1f} ms '
          f'{peak:6.2f} GB')


if __name__ == '__main__':
    main()
