"""Tests of rational_cnn.salt_encoder (need the `salt` package, see
salt_vertexing/README.md)."""
import pytest
import torch

salt_models = pytest.importorskip('salt.models')
from rational_cnn.salt_encoder import GeometricPreEncoder, knn_edges  # noqa

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
B, L, C = 4, 12, 32


def make_inputs():
    torch.manual_seed(0)
    x = torch.randn(B, L, C, device=DEV)
    raw = torch.randn(B, L, 9, device=DEV)
    pad = torch.zeros(B, L, dtype=torch.bool, device=DEV)
    pad[0, 5:] = True
    pad[1, 1:] = True  # a single-track jet: no edges at all
    return x, raw, pad


def make(conv):
    enc = salt_models.Transformer(embed_dim=C, num_layers=1, out_dim=C,
                                  attn_type='torch-math',
                                  attn_kwargs={'num_heads': 4})
    m = GeometricPreEncoder(enc, C, [3, 2, 0, 1], [1, 1, .01, .01],
                            conv=conv, knn_idx=[0, 1], k=4, hidden_dim=16,
                            num_bases=6)
    return m.to(DEV), enc


def test_knn_edges():
    _, raw, pad = make_inputs()
    ei = knn_edges(raw[..., [3, 2]], pad, 4)
    assert not pad.view(-1)[ei].any()
    assert (ei[0] // L == ei[1] // L).all()  # never across jets
    assert (ei[0] != ei[1]).all()
    # jet 0 has 5 tracks -> 4 neighbours each; jet 1 none; jets 2, 3: 12 x 4
    assert ei.size(1) == 5 * 4 + 2 * 12 * 4


@pytest.mark.parametrize('conv', ['none', 'spline', 'rational', 'mlp'])
def test_identity_at_init_and_backward(conv):
    x, raw, pad = make_inputs()
    m, enc = make(conv)
    out, _ = m({'tracks': x}, {'tracks': pad}, inputs={'tracks': raw})
    base, _ = enc({'tracks': x}, {'tracks': pad}, inputs={'tracks': raw})
    assert torch.allclose(out, base, atol=1e-5)
    if conv == 'none':
        return
    torch.nn.init.normal_(m.lin_out.weight, std=0.1)
    with torch.autocast(DEV, dtype=torch.bfloat16 if DEV == 'cpu'
                        else torch.float16):
        out, _ = m({'tracks': x}, {'tracks': pad}, inputs={'tracks': raw})
    assert torch.isfinite(out).all()
    out.float().sum().backward()
    g = m.convs[0].weight.grad
    assert g is not None and torch.isfinite(g).all() and g.norm() > 0
