"""Tests of vertexing/rational_gnn_multicut.py and
vertexing/transformer_edge_multicut.py (shared: data, clustering, metrics)."""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import rational_gnn_multicut as rgnn  # noqa
import transformer_edge_multicut as tedge  # noqa

MODULES = [rgnn, tedge]
DATA = ROOT.parent / 'data' / 'vertexing'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(scope='module')
def batch():
    path = DATA / 'pp_output_val.h5'
    if not path.exists():
        pytest.skip('dataset not downloaded')
    import h5py
    import yaml
    f = h5py.File(path, 'r')
    norm = yaml.safe_load(open(DATA / 'norm_dict.yaml'))
    return rgnn.to_tensors(f['consts'][:2000], f['jets'][:2000],
                           f['hadrons'][:2000], norm)


@pytest.mark.parametrize('mod', MODULES)
def test_gaec_and_components(mod):
    # two attractive cliques {0, 1, 2}, {3, 4}, repulsive in between
    lab = np.array([0, 0, 0, 1, 1])
    cost = np.where(lab[:, None] == lab[None], 2.0, -3.0)
    for solve in (mod.gaec, mod.components):
        out = solve(cost)
        assert (out[:, None] == out[None]).tolist() == \
            (lab[:, None] == lab[None]).tolist()
    assert len(set(mod.gaec(-np.ones((4, 4))))) == 4
    # GAEC respects the repulsive edge that connected components ignore
    cost = np.array([[0, 5, 5], [5, 0, -20.], [5, -20., 0]])
    assert len(set(mod.components(cost))) == 1
    assert len(set(mod.gaec(cost))) == 2


@pytest.mark.parametrize('mod', MODULES)
def test_metrics_oracle(mod, batch):
    # perfect edge scores (same hadron) and an oracle origin (heavy flavour
    # iff linked to a truth hadron; ~10 % of fromB/fromBC/fromC tracks are not)
    # must reconstruct every truth vertex perfectly, without fakes
    same = mod.edge_targets(batch, 'hadron')
    logits = torch.where(same, 5.0, -5.0)
    oracle = torch.where(batch['hadron'] >= 0, mod.HF[0], 0)
    origin = torch.nn.functional.one_hot(oracle, len(mod.ORIGINS)).float()
    for solver in ('multicut', 'cc'):
        acc = Counter(n_jets=len(logits))
        mod.cluster_metrics(logits, origin, batch, solver, 0.0, acc)
        res = mod.summarise(acc)
        assert res['perfect_eff'] == 1.0 and res['loose_eff'] == 1.0
        assert res['perfect_eff_b'] == 1.0 and res['perfect_eff_c'] == 1.0
        assert res['perfect_fake'] == 0.0


def test_pair_geometry_symmetry(batch):
    u = rgnn.pair_geometry(batch['geo'], [0.1, 0.1, 0.5, 0.1, 0.1])
    m = rgnn.pair_mask(batch['mask'])
    assert ((u >= 0) & (u <= 1)).all()
    ut = u.transpose(1, 2)
    for k in (0, 1, 3, 4):  # antisymmetric differences: u_ij + u_ji = 1
        assert torch.allclose((u[..., k] + ut[..., k])[m], torch.tensor(1.),
                              atol=1e-5)
    assert torch.allclose(u[..., 2][m], ut[..., 2][m], atol=1e-4)  # L_ij


def small_args(mod):
    p = dict(hidden=16, layers=1)
    if mod is rgnn:
        p.update(num_bases=4, kernel_size=2, degrees=[2, 1], init='pca',
                 rank=4, scales=[0.1, 0.1, 0.5, 0.1, 0.1])
    else:
        p.update(heads=2)
    return argparse.Namespace(edge_label='vertex', **p)


@pytest.mark.parametrize('mod', MODULES)
def test_forward_backward(mod, batch):
    args = small_args(mod)
    model = mod.build_model(args).to(DEV)
    b = {k: v[:64] for k, v in batch.items()}
    L = int(b['mask'].sum(1).max())
    b = {k: v[:, :L] if v.dim() > 1 and v.size(1) == batch['mask'].size(1)
         else v for k, v in b.items()}
    weights = torch.ones(len(mod.ORIGINS), device=DEV)
    bd, logits, origin = mod.forward_batch(model, b, args, torch.device(DEV))
    assert logits.shape == (64, L, L)
    assert torch.allclose(logits, logits.transpose(1, 2))
    edge, orig, _ = mod.losses(bd, logits, origin, args, weights)
    (edge + orig).backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
