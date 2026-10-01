r"""Geometric graph-convolution pre-encoder for SALT (ATLAS flavour-tagging
framework, https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt).

:class:`GeometricPreEncoder` wraps any SALT encoder (typically
:class:`salt.models.Transformer`). Before the wrapped encoder runs, it builds
a k-nearest-neighbour graph over the tracks of every jet and applies a stack
of continuous-kernel graph convolutions (SplineConv with B-spline hats, or
RationalConv with learnable rational bases) whose pseudo-coordinates are the
pairwise *differences* of geometric track variables (e.g. eta, phi, d0, z0).
The result is added to the track embeddings through a zero-initialised
projection, so at initialisation the model is exactly the wrapped encoder.

Single file of the MaskFormer approach: the pre-encoder and a callback
writing the epoch metrics to ``metrics.csv`` (SALT's CLI only wires up a Comet
logger). Config: ``vertexing/configs/maskformer_geo.yaml``; SALT imports this
module as ``maskformer_geo``, so run with ``PYTHONPATH=vertexing``.
"""
import csv
from pathlib import Path

import torch
from torch import nn

from lightning import Callback

from rational_cnn import BSplineConv, RationalConv


def knn_edges(coords, pad_mask, k):
    r"""Per-jet kNN graph on padded tensors.

    Args:
        coords: ``[B, L, D]`` coordinates the neighbourhood is built in.
        pad_mask: ``[B, L]`` (``True`` = padded track) or ``None``.
        k: neighbours per track (fewer if a jet has fewer tracks).

    Returns:
        ``edge_index`` of shape ``[2, E]`` (source ``j``, target ``i``) into the
        flattened ``B * L`` track list; padded tracks have no edges.
    """
    B, L, _ = coords.shape
    k = min(k, L - 1)
    dist = torch.cdist(coords, coords)  # [B, L, L]
    eye = torch.eye(L, dtype=torch.bool, device=coords.device)
    invalid = eye.expand(B, L, L)
    if pad_mask is not None:
        invalid = invalid | pad_mask[:, None, :] | pad_mask[:, :, None]
    dist = dist.masked_fill(invalid, float('inf'))
    d, nbr = dist.topk(k, dim=-1, largest=False)  # [B, L, k]
    keep = torch.isfinite(d)
    offset = (torch.arange(B, device=coords.device) * L).view(B, 1, 1)
    tgt = (torch.arange(L, device=coords.device).view(1, L, 1) + offset)
    src = nbr + offset
    tgt = tgt.expand_as(src)
    return torch.stack([src[keep], tgt[keep]], dim=0)


class GeometricPreEncoder(nn.Module):
    r"""SplineConv / RationalConv GNN prepended to a SALT encoder.

    Args:
        encoder: The wrapped SALT encoder (called with the updated embeddings
            and all other arguments unchanged).
        embed_dim: Width of the track embeddings produced by the init net.
        coord_idx: Columns of ``inputs[input_name]`` (normalised track
            variables, in the order of the ``data.variables`` config) used as
            geometric coordinates.
        coord_scale: Per-coordinate scale ``s_d`` (in normalised units) of the
            pseudo-coordinate squashing
            ``u_d = 1/2 + 1/2 tanh(asinh((c_{j,d} - c_{i,d}) / s_d) / 2)``
            (the asinh tames the heavy tails of impact-parameter
            differences).
            (default: ``1.0`` for every coordinate)
        conv: ``'spline'`` (B-spline hats, SplineCNN), ``'rational'``
            (multivariate rational basis) or ``'none'`` (identity, i.e. the
            plain wrapped encoder).
        hidden_dim: Width of the graph convolutions.
        num_layers: Number of graph convolutions (each residual).
        k: kNN neighbours per track.
        knn_idx: Coordinates (indices into ``coord_idx``) the kNN graph is
            built in. (default: all)
        kernel_size: B-spline knots per dimension (also the hat grid the
            rational ``'spline'``/``'pca'`` initialisations refer to).
        num_bases: Number of rational basis functions (``None`` =
            ``kernel_size ** D``).
        degrees: Numerator / denominator total degrees of the rational basis.
        init: Rational basis initialisation (``'pca'``, ``'spline'``,
            ``'random'``, ...; see :class:`MultivariateRationalBasis`).
        aggr: Neighbourhood aggregation.
        input_name: Input modality carrying the tracks.
    """
    def __init__(self, encoder: nn.Module, embed_dim: int,
                 coord_idx: list[int], coord_scale: list[float] | None = None,
                 conv: str = 'rational', hidden_dim: int = 64,
                 num_layers: int = 2, k: int = 8,
                 knn_idx: list[int] | None = None, kernel_size: int = 3,
                 num_bases: int | None = None, degrees: list[int] = (4, 3),
                 init: str = 'pca', aggr: str = 'mean',
                 input_name: str = 'tracks'):
        super().__init__()
        assert conv in ['spline', 'rational', 'none']
        self.encoder = encoder
        self.conv_type = conv
        self.input_name = input_name
        self.k = k
        D = len(coord_idx)
        self.register_buffer('coord_idx', torch.tensor(coord_idx),
                             persistent=False)
        scale = torch.tensor(coord_scale if coord_scale is not None
                             else [1.0] * D, dtype=torch.float)
        assert scale.numel() == D
        self.register_buffer('coord_scale', scale, persistent=False)
        self.knn_idx = list(knn_idx) if knn_idx is not None else list(range(D))
        if conv == 'none':
            return

        self.lin_in = nn.Linear(embed_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            if conv == 'spline':
                c = BSplineConv(hidden_dim, hidden_dim, D, kernel_size,
                                aggr=aggr)
            else:
                c = RationalConv(hidden_dim, hidden_dim, D, kernel_size,
                                 basis='multivariate', num_bases=num_bases,
                                 degrees=tuple(degrees), init=init, aggr=aggr)
            self.convs.append(c)
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.lin_out = nn.Linear(hidden_dim, embed_dim)
        nn.init.zeros_(self.lin_out.weight)
        nn.init.zeros_(self.lin_out.bias)

    def geometric(self, x, raw, pad_mask):
        r"""The GNN residual for track embeddings ``x`` ``[B, L, C]``."""
        B, L, C = x.shape
        coords = raw[..., self.coord_idx].float()
        coords = torch.nan_to_num(coords)
        edge_index = knn_edges(coords[..., self.knn_idx], pad_mask, self.k)
        src, tgt = edge_index
        flat = coords.reshape(B * L, -1)
        delta = (flat[src] - flat[tgt]) / self.coord_scale
        pseudo = 0.5 + 0.5 * torch.tanh(torch.asinh(delta) / 2)

        h = self.lin_in(x.float()).reshape(B * L, -1)
        for conv, norm in zip(self.convs, self.norms):
            h = h + conv(nn.functional.relu(norm(h)), edge_index, pseudo)
        out = self.lin_out(h).view(B, L, C)
        if pad_mask is not None:
            out = out.masked_fill(pad_mask[..., None], 0.0)
        return out

    def forward(self, x, pad_mask, inputs=None, **kwargs):
        if self.conv_type != 'none':
            name = self.input_name
            mask = pad_mask[name] if isinstance(pad_mask, dict) else pad_mask
            xt = x[name] if isinstance(x, dict) else x
            # Bases and kNN in full precision (the rational denominators and
            # the tanh squashing are not fp16-safe).
            with torch.autocast(device_type=xt.device.type, enabled=False):
                res = self.geometric(xt, inputs[name], mask)
            xt = xt + res.to(xt.dtype)
            if isinstance(x, dict):
                x = {**x, name: xt}
            else:
                x = xt
        return self.encoder(x, pad_mask, inputs=inputs, **kwargs)


class MetricsCSV(Callback):
    r"""Appends one row of ``trainer.callback_metrics`` per validation epoch to
    ``<default_root_dir>/metrics.csv``."""
    def __init__(self, filename: str = 'metrics.csv'):
        self.filename = filename
        self.rows = []

    def on_validation_epoch_end(self, trainer, module):
        if trainer.sanity_checking or trainer.global_rank != 0:
            return
        row = {'epoch': trainer.current_epoch, 'step': trainer.global_step}
        for k, v in trainer.callback_metrics.items():
            row[k] = float(v)
        self.rows.append(row)
        keys = sorted({k for r in self.rows for k in r},
                      key=lambda k: (k not in ('epoch', 'step'), k))
        path = Path(trainer.default_root_dir) / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)
