r"""Out-of-task transfer of a continuous-kernel keypoint refinement, measured
on HPatches.

The question. The graph convolution of NMT (and of DGMC) is trained to refine a
keypoint's descriptor from its neighbours and their relative positions, for
*semantic* matching: the same body part across two different bicycles. Does
that refinement, frozen, also help *geometric* matching -- the same physical
scene seen from another viewpoint -- which it never saw during training? If it
does, the operator learned a reusable geometric prior rather than a task.

The protocol ("projection"). For an HPatches pair, N keypoints are detected in
the reference image and mapped into the target image with the ground-truth
homography, so the correct correspondence is known by construction and is a
permutation of the N candidates. A matcher then picks, for every reference
keypoint, one of the N projected candidates. Accuracy is therefore a pure
descriptor-quality measurement: no detector repeatability, no ratio test, no
RANSAC and no reconstruction. Errors are reported in the target image's own
pixels, as mean matching accuracy at 1/3/5 px.

What varies between arms is only how a keypoint's descriptor is produced; the
backbone, the keypoints and the graphs are identical. See
experiments/hpatches_transfer.py.
"""
import glob
import json
import os.path as osp
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data

from .bspline import BSplineConv
from .face_to_edge import FaceToEdge
from .rational import RationalConv

#: The resolution NMT trains at; keypoint features have to stay in-distribution.
SIZE = 256
#: Delaunay graph with Cartesian pseudo-coordinates, as in the experiments.
TRANSFORM = T.Compose([T.Delaunay(), FaceToEdge(), T.Cartesian()])


# ------------------------------------------------------------------ dataset
def sequences(root: str):
    r"""The 116 HPatches sequences as ``(name, kind)``, kind being
    ``"illumination"`` (``i_``) or ``"viewpoint"`` (``v_``)."""
    out = []
    for path in sorted(glob.glob(osp.join(root, '*'))):
        name = osp.basename(path)
        if not osp.isdir(path) or not name[:2] in ('i_', 'v_'):
            continue
        out.append((name, 'illumination' if name[0] == 'i' else 'viewpoint'))
    return out


def load_pair(root: str, name: str, target: int):
    r"""Reference image 1, target image ``target`` (2..6) and the homography
    mapping reference pixels to target pixels."""
    from PIL import Image
    ref = Image.open(osp.join(root, name, '1.ppm')).convert('RGB')
    trg = Image.open(osp.join(root, name, f'{target}.ppm')).convert('RGB')
    H = np.loadtxt(osp.join(root, name, f'H_1_{target}'))
    return ref, trg, H


def project(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    r"""Applies a homography to ``[N, 2]`` pixel coordinates."""
    p = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    q = p @ np.asarray(H, dtype=np.float64).T
    return q[:, :2] / q[:, 2:3]


# ----------------------------------------------------------------- detector
def shi_tomasi(gray: torch.Tensor, num_keypoints: int, border: int = 16,
               window: int = 5, nms: int = 9) -> torch.Tensor:
    r"""Shi-Tomasi corners of a ``[1, 1, H, W]`` grayscale image in ``[0, 1]``,
    as ``[N, 2]`` ``(x, y)`` coordinates ordered by decreasing response.

    Implemented here in pure PyTorch so that this experiment adds no dependency
    to the project's environment. The detector only decides *where* descriptors
    are sampled -- correspondences come from the homography -- so its exact
    choice is not what is being measured.
    """
    kernel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                          device=gray.device).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kernel, padding=1)
    gy = F.conv2d(gray, kernel.transpose(-1, -2), padding=1)

    box = torch.ones(1, 1, window, window, device=gray.device) / window ** 2
    a = F.conv2d(gx * gx, box, padding=window // 2)
    b = F.conv2d(gy * gy, box, padding=window // 2)
    c = F.conv2d(gx * gy, box, padding=window // 2)
    # smaller eigenvalue of the structure tensor
    root = torch.sqrt(torch.clamp(((a - b) / 2) ** 2 + c ** 2, min=0))
    response = (a + b) / 2 - root

    response[..., :border, :] = -1
    response[..., -border:, :] = -1
    response[..., :, :border] = -1
    response[..., :, -border:] = -1

    pooled = F.max_pool2d(response, nms, stride=1, padding=nms // 2)
    response = torch.where((response == pooled) & (response > 0), response,
                           torch.full_like(response, -1))

    flat = response.view(-1)
    n = int(min(num_keypoints, int((flat > 0).sum())))
    if n == 0:
        return torch.zeros(0, 2)
    idx = torch.topk(flat, n).indices
    width = gray.size(-1)
    return torch.stack([(idx % width), (idx // width)], dim=1).float().cpu()


def distractors(gray: torch.Tensor, avoid: np.ndarray, num: int,
                min_distance: float = 6.0) -> torch.Tensor:
    r"""Keypoints detected independently in the target image, excluding any
    that fall within :obj:`min_distance` of a ground-truth correspondence.

    Without these the candidate set is exactly the homography image of the
    query set, the two Delaunay graphs are near-isomorphic, and a matcher can
    succeed on topology alone -- measured at 88% with constant (appearance-free)
    features. Mixing in independently detected points removes that shortcut.
    """
    if num <= 0:
        return torch.zeros(0, 2)
    points = shi_tomasi(gray, max(4 * num, num + 32))
    if len(points) and len(avoid):
        keep = torch.cdist(points, torch.as_tensor(avoid, dtype=torch.float))
        points = points[keep.min(dim=1).values > min_distance]
    return points[:num]


def assemble_candidates(gt_points: np.ndarray, extra: torch.Tensor,
                        seed: int):
    r"""Mixes the true correspondences and the distractors into one shuffled
    candidate set, and returns it together with the index of each query's
    correct candidate.

    Shuffling matters: without it the answer is always index ``i``, which both
    leaks the assignment and lets argmax ties score as correct.
    """
    candidates = (np.concatenate([gt_points, extra.numpy()], axis=0)
                  if len(extra) else np.asarray(gt_points))
    perm = torch.randperm(len(candidates),
                          generator=torch.Generator().manual_seed(int(seed)))
    candidates = candidates[perm.numpy()]
    # position[i] is where the originally i-th entry ended up
    position = torch.empty(len(candidates), dtype=torch.long)
    position[perm] = torch.arange(len(candidates))
    return candidates, position[:len(gt_points)]


def to_gray(image) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
    return x.mean(-1).view(1, 1, *x.shape[:2])


# ------------------------------------------------------------------ backbone
def _vgg16_parts():
    r"""VGG16 split at relu4_2 and relu5_1, exactly as NMT's
    ``VGG16_base.get_backbone``. Weights are replaced from a checkpoint."""
    from torchvision import models
    node, edge, current = [], [], []
    cnt_m, cnt_r = 1, 0
    for module in models.vgg16().features.children():
        if isinstance(module, torch.nn.Conv2d):
            cnt_r += 1
        if isinstance(module, torch.nn.MaxPool2d):
            cnt_r, cnt_m = 0, cnt_m + 1
        current.append(module)
        if cnt_m == 4 and cnt_r == 2 and isinstance(module, torch.nn.ReLU):
            node, current = current, []
        elif cnt_m == 5 and cnt_r == 1 and isinstance(module, torch.nn.ReLU):
            edge, current = current, []
    return torch.nn.Sequential(*node), torch.nn.Sequential(*edge)


def normalize_over_channels(x, eps=1e-9):
    return x / (torch.norm(x, dim=1, keepdim=True) + eps)


class KeypointFeatures(torch.nn.Module):
    r"""The frozen VGG16 of an NMT checkpoint, sampled at keypoints exactly as
    NMT does it: ``relu4_2`` and ``relu5_1`` are normalised over channels and
    read out bilinearly, giving 512 + 512 = 1024 channels per keypoint."""
    def __init__(self, state_dict: Optional[dict] = None):
        super().__init__()
        self.node_layers, self.edge_layers = _vgg16_parts()
        if state_dict is not None:
            load = {k: v for k, v in state_dict.items()
                    if k.startswith(('node_layers.', 'edge_layers.'))}
            missing = self.load_state_dict(load, strict=False)
            assert not missing.unexpected_keys, missing.unexpected_keys
        self.eval()

    @torch.no_grad()
    def forward(self, image: torch.Tensor, points: torch.Tensor):
        r"""``image`` is ``[1, 3, SIZE, SIZE]`` normalised for ImageNet,
        ``points`` are ``[N, 2]`` pixel coordinates in that resized image."""
        nodes = normalize_over_channels(self.node_layers(image))
        edges = normalize_over_channels(self.edge_layers(nodes))
        grid = (points / SIZE * 2 - 1).view(1, 1, -1, 2).to(image.device)
        out = []
        for feat in (nodes, edges):
            sampled = F.grid_sample(feat, grid, mode='bilinear',
                                    padding_mode='border', align_corners=False)
            out.append(sampled.squeeze(0).squeeze(1).t())
        return torch.cat(out, dim=-1)


# ---------------------------------------------------------------- refinement
def make_conv(cfg: dict, in_channels: int, out_channels: int):
    r"""One continuous-kernel convolution from an NMT ``SPLINE_CNN`` config."""
    get = cfg.get
    common = dict(aggr=get('aggr', 'max'), pyg_init=get('pyg_init', True))
    if get('conv', 'spline') == 'spline':
        return BSplineConv(in_channels, out_channels, get('dim', 2),
                           get('kernel_size', 5), **common)
    return RationalConv(in_channels, out_channels, get('dim', 2),
                        get('kernel_size', 5),
                        basis=get('rational_basis', 'product'),
                        vp=get('vp', False),
                        degrees=tuple(get('degrees', [5, 4])),
                        safe=get('safe', 'B'), poly=get('poly', 'chebyshev'),
                        init=get('init', 'spline'),
                        init_noise=get('init_noise', 1e-3),
                        num_bases=get('num_bases', 0) or None, **common)


class Refinement(torch.nn.Module):
    r"""NMT's per-keypoint refinement: two continuous-kernel convolutions with
    a ReLU between them, added to a linear projection of the raw descriptor,

    .. math:: h = \psi(x, \mathcal{G}) + W x,

    which is the representation NMT hands to its matching transformer. Only the
    basis inside :math:`\psi` differs between the spline and rational arms.
    """
    def __init__(self, spline_cfg: dict, in_channels: int = 1024,
                 out_channels: int = 648):
        super().__init__()
        self.conv_0 = make_conv(spline_cfg, in_channels, out_channels)
        self.conv_1 = make_conv(spline_cfg, out_channels, out_channels)
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.eval()

    @torch.no_grad()
    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.conv_0(x, edge_index, edge_attr))
        h = self.conv_1(h, edge_index, edge_attr)
        return h + self.lin(x)

    @staticmethod
    def from_checkpoint(run_dir: str, randomize: bool = False):
        r"""Builds the refinement of an NMT run and loads its trained weights.
        With :obj:`randomize`, the architecture and configuration are identical
        but the weights stay at their initialisation -- the control that
        separates a learned prior from the shape of the operator."""
        with open(osp.join(run_dir, 'settings.json')) as f:
            settings = json.load(f)
        cfg = settings['SPLINE_CNN']
        module = Refinement(cfg, cfg['input_features'],
                            settings['Matching_TF']['d_model'])
        if randomize:
            return module, settings
        state = load_state(run_dir)
        load = {}
        for key, value in state.items():
            if key.startswith('psi.convs.0.'):
                load['conv_0.' + key[len('psi.convs.0.'):]] = value
            elif key.startswith('psi.convs.1.'):
                load['conv_1.' + key[len('psi.convs.1.'):]] = value
            elif key.startswith('vit_to_node_dim.'):
                load['lin.' + key[len('vit_to_node_dim.'):]] = value
        out = module.load_state_dict(load, strict=False)
        assert not out.unexpected_keys, out.unexpected_keys
        assert not [k for k in out.missing_keys
                    if not k.startswith(('conv_0.basis', 'conv_1.basis'))], \
            out.missing_keys
        return module, settings


def load_state(run_dir: str) -> dict:
    r"""The newest epoch checkpoint of an NMT run, with the DDP prefix removed."""
    epochs = sorted(glob.glob(osp.join(run_dir, 'params', '[0-9]' * 4)))
    if not epochs:
        raise FileNotFoundError(f'no checkpoint under {run_dir}/params')
    state = torch.load(osp.join(epochs[-1], 'params.pt'), map_location='cpu',
                       weights_only=False)
    return {k[len('module.'):] if k.startswith('module.') else k: v
            for k, v in state.items()}


class MeanAggregation(torch.nn.Module):
    r"""Control arm: replace the learned convolution by the mean of the
    Delaunay neighbours, keeping NMT's residual weight. If this matches the
    trained module, the gain was smoothing rather than anything learned."""
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    @torch.no_grad()
    def forward(self, x, edge_index, edge_attr=None):
        out = torch.zeros_like(x)
        count = torch.zeros(x.size(0), 1, device=x.device)
        src, dst = edge_index
        out.index_add_(0, dst, x[src])
        count.index_add_(0, dst, torch.ones_like(count[dst]))
        return x + self.weight * out / count.clamp(min=1)


# -------------------------------------------------------------------- graph
def build_graph(points: torch.Tensor):
    r"""Delaunay triangulation with Cartesian pseudo-coordinates."""
    data = TRANSFORM(Data(pos=points.clone()))
    edge_attr = data.edge_attr
    if edge_attr is not None:
        edge_attr = torch.nan_to_num(edge_attr, nan=0.5)
    return data.edge_index, edge_attr


# ------------------------------------------------------------------- metric
@torch.no_grad()
def match(features_ref: torch.Tensor, features_trg: torch.Tensor):
    r"""Cosine-similarity assignment: for every reference keypoint the index of
    the most similar target candidate. No learned matcher is involved, so the
    only thing that differs between arms is the descriptor."""
    a = F.normalize(features_ref.float(), dim=-1)
    b = F.normalize(features_trg.float(), dim=-1)
    return (a @ b.t()).argmax(dim=-1)


def score(pred: torch.Tensor, candidates: np.ndarray,
          gt_index: torch.Tensor, thresholds=(1.0, 3.0, 5.0)) -> dict:
    r"""Exact matching accuracy plus mean matching accuracy at pixel
    thresholds, in the target image's own resolution.

    Args:
        pred: the chosen candidate per query, into :obj:`candidates`.
        candidates: ``[n + m, 2]`` target locations, ground truth and
            distractors together, in target pixels.
        gt_index: the correct candidate per query.
    """
    chosen = torch.as_tensor(candidates[pred.cpu().numpy()])
    truth = torch.as_tensor(candidates[gt_index.cpu().numpy()])
    error = torch.linalg.norm(chosen - truth, dim=-1)
    out = {'queries': int(len(gt_index)), 'candidates': int(len(candidates)),
           'acc': float((pred.cpu() == gt_index).float().mean())}
    for t in thresholds:
        out[f'mma@{t:g}'] = float((error <= t).float().mean())
    return out
