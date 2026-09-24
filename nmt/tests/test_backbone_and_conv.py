r"""Tests for the two changes that make NMT comparable to the DGMC setup:

* the visual backbone can be the plain VGG16 of SplineCNN/DGMC instead of
  SwinV2-L, and
* the continuous-kernel convolution can use learnable rational basis functions
  instead of B-spline hats.

They run on CPU and need no dataset; the VGG16 test does load the ImageNet
weights, which hpc/create_environment.sh pre-fetches into $TORCH_HOME.
"""
import json
import os
import os.path as osp

import numpy as np
import pytest
import torch
from PIL import Image
from torch_geometric.data import Data

from rational_cnn import BSplineConv, RationalConv
from utils.config import cfg


def small_graph(num_nodes=6, channels=16, num_edges=14):
    torch.manual_seed(0)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return Data(x=torch.randn(num_nodes, channels), edge_index=edge_index,
                edge_attr=torch.rand(num_edges, 2))


# --------------------------------------------------------------- backbone
def test_vgg16_backbone_shapes(monkeypatch):
    monkeypatch.setitem(cfg, 'BACKBONE', 'vgg16')
    from utils.backbone import VisualBackbone

    backbone = VisualBackbone()
    assert backbone.backbone_name == 'vgg16'

    image = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        nodes, edges, glob = backbone.extract(image)

    # relu4_2 after three poolings, relu5_1 after four, 512 channels each.
    assert nodes.shape == (2, 512, 32, 32)
    assert edges.shape == (2, 512, 16, 16)
    # The global vector must be cfg.SPLINE_CNN.input_features // 2 wide, since
    # NMT derives the global projection from it.
    assert glob.shape == (2, 512)
    assert 512 + 512 == 1024

    # train_eval.py puts these in an optimizer group at 0.03x the base LR.
    assert len(backbone.backbone_params) > 0
    owned = {id(p) for p in backbone.parameters()}
    assert all(id(p) in owned for p in backbone.backbone_params)


def test_unknown_backbone_is_rejected(monkeypatch):
    monkeypatch.setitem(cfg, 'BACKBONE', 'resnet50')
    from utils.backbone import VisualBackbone
    with pytest.raises(ValueError, match='unknown cfg.BACKBONE'):
        VisualBackbone()


# ------------------------------------------------------------------- conv
def test_make_conv_spline(monkeypatch):
    monkeypatch.setitem(cfg.SPLINE_CNN, 'conv', 'spline')
    from model.sconv_archs import make_conv

    conv = make_conv(16, 8)
    assert isinstance(conv, BSplineConv)
    # kernel_size 5 over 2-D pseudo-coordinates = 25 B-spline hats
    assert conv.weight.shape == (25, 16, 8)

    data = small_graph()
    out = conv(data.x, data.edge_index, data.edge_attr)
    assert out.shape == (6, 8) and torch.isfinite(out).all()


def test_make_conv_rational(monkeypatch):
    for key, value in dict(conv='rational', rational_basis='multivariate',
                           degrees=[8, 6], init='pca', vp=True,
                           num_bases=4).items():
        monkeypatch.setitem(cfg.SPLINE_CNN, key, value)
    from model.sconv_archs import make_conv

    conv = make_conv(16, 8)
    assert isinstance(conv, RationalConv)
    # K is decoupled from the grid: 4 basis functions instead of 5**2 = 25.
    assert conv.weight.shape == (4, 16, 8)

    data = small_graph()
    out = conv(data.x, data.edge_index, data.edge_attr)
    assert out.shape == (6, 8) and torch.isfinite(out).all()


def test_free_num_bases_requires_pca_init(monkeypatch):
    """K != kernel_size**dim cannot use the hat-fitting initialisation, because
    that fits one rational function per B-spline hat. The configs therefore
    pair num_bases with init='pca'; getting this wrong must fail loudly at
    construction rather than silently training something else."""
    for key, value in dict(conv='rational', rational_basis='multivariate',
                           init='spline', num_bases=4).items():
        monkeypatch.setitem(cfg.SPLINE_CNN, key, value)
    from model.sconv_archs import make_conv
    with pytest.raises(AssertionError, match='num_bases'):
        make_conv(16, 8)


def test_unknown_conv_is_rejected(monkeypatch):
    monkeypatch.setitem(cfg.SPLINE_CNN, 'conv', 'gcn')
    from model.sconv_archs import make_conv
    with pytest.raises(ValueError, match='unknown cfg.SPLINE_CNN.conv'):
        make_conv(16, 8)


@pytest.mark.parametrize('conv_type', ['spline', 'rational'])
def test_sconv_forward_and_backward(monkeypatch, conv_type):
    monkeypatch.setitem(cfg.SPLINE_CNN, 'conv', conv_type)
    monkeypatch.setitem(cfg.SPLINE_CNN, 'rational_basis', 'multivariate')
    # A free K needs the PCA-of-hats initialisation; see the guard test below.
    monkeypatch.setitem(cfg.SPLINE_CNN, 'init', 'pca')
    monkeypatch.setitem(cfg.SPLINE_CNN, 'vp', True)
    monkeypatch.setitem(cfg.SPLINE_CNN, 'num_bases', 4)
    from model.sconv_archs import SConv, SiameseSConvOnNodes

    model = SConv(input_features=16, output_features=8)
    assert model.num_layers == 2 and model.out_channels == 8

    data = small_graph()
    out = model(data)
    assert out.shape == (6, 8) and torch.isfinite(out).all()
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)

    # the residual wrapper NMT actually uses keeps the feature dimension
    siamese = SiameseSConvOnNodes(input_node_dim=16)
    graph = siamese(small_graph())
    assert graph.x.shape == (6, 16)


# ---------------------------------------------------- edge pseudo-coordinates
def test_build_graphs_scales_with_image_size(monkeypatch):
    from utils.build_graphs import build_graphs

    points = np.array([[0.0, 0.0], [200.0, 10.0], [10.0, 200.0], [90.0, 90.0]])

    monkeypatch.setitem(cfg, 'IMAGE_SIZE', 256)
    edges, feat256 = build_graphs(points, 4)
    assert edges.shape[0] == 2 and feat256.shape[1] == 2
    # SplineConv requires pseudo-coordinates inside [0, 1].
    assert feat256.min() >= 0.0 and feat256.max() <= 1.0
    row, col = edges[0][0], edges[1][0]
    expected = 0.5 + 0.5 * (points[row, 0] - points[col, 0]) / 256.0
    assert abs(feat256[0, 0] - expected) < 1e-12

    # The same keypoints must give different pseudo-coordinates at 384, which
    # is why cfg.IMAGE_SIZE and the dataset resize have to agree.
    monkeypatch.setitem(cfg, 'IMAGE_SIZE', 384)
    _, feat384 = build_graphs(points, 4)
    assert not np.allclose(feat256, feat384)


# ------------------------------------------------------- SPair-71k filenames
def write_json(path, obj):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f)


def make_spair(root, separator):
    cat = 'aeroplane'
    for name in ('a', 'b'):
        path = osp.join(root, 'JPEGImages', cat, f'{name}.jpg')
        os.makedirs(osp.dirname(path), exist_ok=True)
        Image.fromarray(np.full((40, 50, 3), 128, dtype=np.uint8)).save(path)
    pair = f'000001-a-b{separator}{cat}'
    write_json(osp.join(root, 'PairAnnotation', 'trn', f'{pair}.json'),
               dict(category=cat, src_imname='a.jpg', trg_imname='b.jpg',
                    src_kps=[[10, 10], [20, 20], [30, 15]],
                    trg_kps=[[12, 11], [22, 21], [33, 16]]))
    layout = osp.join(root, 'Layout', 'large', 'trn.txt')
    os.makedirs(osp.dirname(layout), exist_ok=True)
    # The Layout files always use ":", whatever the annotation files use.
    with open(layout, 'w') as f:
        f.write(f'000001-a-b:{cat}\n')


@pytest.mark.parametrize('separator', [':', '_'])
def test_spair_finds_pair_annotations(tmp_path, monkeypatch, separator):
    root = str(tmp_path / 'SPair-71k')
    make_spair(root, separator)
    monkeypatch.setitem(cfg.SPair, 'ROOT_DIR', root)
    monkeypatch.setitem(cfg.SPair, 'size', 'large')
    from data.SPair71k import SPair71k

    dataset = SPair71k('train', (256, 256))
    assert dataset.ann_files == [f'000001-a-b{separator}aeroplane']

    anno_list, perm_mats = dataset.get_k_samples(0, k=2, mode='intersection',
                                                cls='aeroplane')
    assert len(anno_list) == 2
    assert perm_mats[0].shape == (3, 3)
    # every source keypoint is matched exactly once
    assert perm_mats[0].sum() == 3
    for anno in anno_list:
        assert anno['image'].size == (256, 256)
        assert len(anno['keypoints']) == 3
