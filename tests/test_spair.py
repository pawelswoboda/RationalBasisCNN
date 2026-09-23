import json
import os
import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from PIL import Image
from torch_geometric.loader import DataLoader

from rational_cnn import DGMC, SplineCNN, FaceToEdge
from rational_cnn.spair import (SPair71k, SPair71kPairs, MatchingAccuracy,
                                CATEGORIES, crop_box)

TRANSFORM = T.Compose([T.Delaunay(), FaceToEdge(), T.Cartesian()])
BOX = [40, 40, 200, 160]
KPS_A = {'0': [60, 60], '1': None, '2': [120, 90], '5': [180, 150]}
KPS_B = {**KPS_A, '7': [100, 130]}  # one more keypoint, still inside BOX
KPS_C = {'2': [70, 100], '5': [230, 20], '0': None}  # (230, 20) is outside BOX


def fake_extractor(images):
    # Two feature maps at VGG16-like reduced resolutions whose values depend
    # on the pixel content, so identical crops give identical features.
    m1 = F.avg_pool2d(images, 8).repeat(1, 171, 1, 1)[:, :512]
    m2 = F.avg_pool2d(images, 16).repeat(1, 171, 1, 1)[:, :512]
    return [m1, m2]


def write_json(path, obj):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f)


def make_spair(root):
    r"""A two-category miniature of the SPair-71k tree. Images a and b have
    identical pixels and share keypoints 0, 2, 5 at identical positions (b
    also has 7), so their common keypoints get identical features."""
    torch.manual_seed(0)
    pixels = (torch.rand(240, 320, 3) * 255).to(torch.uint8).numpy()
    other = (torch.rand(240, 320, 3) * 255).to(torch.uint8).numpy()
    for cat in ('aeroplane', 'bicycle'):
        for name, kps, px in (('a', KPS_A, pixels), ('b', KPS_B, pixels),
                              ('c', KPS_C, other)):
            path = osp.join(root, 'JPEGImages', cat, f'{name}.jpg')
            os.makedirs(osp.dirname(path), exist_ok=True)
            Image.fromarray(px).save(path, format='PNG')  # lossless
            write_json(osp.join(root, 'ImageAnnotation', cat, f'{name}.json'),
                       dict(filename=f'{name}.jpg', category=cat, bndbox=BOX,
                            kps=kps))
        pairs = {'trn': ('a', 'b', ['0', '2', '5']),
                 'val': ('a', 'c', ['2', '5']),
                 'test': ('b', 'c', ['2', '5'])}
        for split, (s, t, ids) in pairs.items():
            name = f'000001-{s}-{t}:{cat}'
            write_json(osp.join(root, 'PairAnnotation', split, f'{name}.json'),
                       dict(category=cat, src_imname=f'{s}.jpg',
                            trg_imname=f'{t}.jpg', kps_ids=ids,
                            viewpoint_variation=1, scale_variation=0,
                            truncation=2, occlusion=0))
    for layout in ('large', 'small'):
        for split in ('trn', 'val', 'test'):
            path = osp.join(root, 'Layout', layout, f'{split}.txt')
            os.makedirs(osp.dirname(path), exist_ok=True)
            cats = ('aeroplane', 'bicycle') if layout == 'large' else (
                'bicycle', )
            names = {'trn': 'a-b', 'val': 'a-c', 'test': 'b-c'}[split]
            with open(path, 'w') as f:
                f.writelines(f'000001-{names}:{c}\n' for c in cats)


def make_dataset(tmp_path, **kwargs):
    raw = str(tmp_path / 'SPair-71k')
    make_spair(raw)
    return SPair71k(str(tmp_path / 'SPair71k'), raw_dir=raw, device='cpu',
                    batch_size=4, extractor=fake_extractor, **kwargs)


def test_spair_processing(tmp_path):
    dataset = make_dataset(tmp_path)
    assert len(dataset) == 6
    assert dataset.num_node_features == 1024
    c = dataset[2]  # aeroplane/c: visible keypoints 2 and 5 only
    assert c.kp.tolist() == [2, 5]
    pos = torch.tensor([[70., 100.], [230., 20.]])
    box = crop_box(pos, BOX)
    assert box == (40 - 16, 20 - 16, 230 + 16, 160 + 16)  # grown to (230, 20)
    expected = torch.stack([(pos[:, 0] - box[0]) * 256 / (box[2] - box[0]),
                            (pos[:, 1] - box[1]) * 256 / (box[3] - box[1])], 1)
    assert torch.allclose(c.pos, expected)

    pairs = dataset.pairs
    for split in ('trn', 'val', 'test'):
        assert pairs['layout']['large'][split].numel() == 2
        assert pairs['layout']['small'][split].numel() == 1
    assert pairs['difficulty']['truncation'].tolist() == [2] * 6
    for p in range(pairs['src'].numel()):  # nodes hold exactly the kps_ids
        lo, hi = int(pairs['ptr'][p]), int(pairs['ptr'][p + 1])
        src = dataset[int(pairs['src'][p])].kp[pairs['src_nodes'][lo:hi]]
        trg = dataset[int(pairs['trg'][p])].kp[pairs['trg_nodes'][lo:hi]]
        assert src.tolist() == trg.tolist()
        assert src.tolist() in ([0, 2, 5], [2, 5])


def test_spair_pairs(tmp_path):
    dataset = make_dataset(tmp_path)
    train = SPair71kPairs(dataset, 'trn', 'large', TRANSFORM, 'random')
    assert str(train) == ('SPair71kPairs(2, split=trn, layout=large, '
                          'shuffle=random)')
    perms = set()
    for _ in range(30):
        pair = train[0]
        assert pair.x_s.size() == (3, 1024) and pair.x_t.size() == (3, 1024)
        assert pair.edge_attr_s.size(-1) == 2
        # a and b have identical features for the common keypoints, so the
        # ground truth must map every source node onto the same feature row.
        assert torch.allclose(pair.x_t[pair.y], pair.x_s)
        assert pair.category == CATEGORIES.index('aeroplane')
        perms.add(tuple(pair.y.tolist()))
    assert len(perms) > 1  # the target order really is shuffled

    fixed = SPair71kPairs(dataset, 'trn', 'large', TRANSFORM, 'fixed')
    assert fixed[1].y.tolist() == fixed[1].y.tolist()
    assert torch.allclose(fixed[1].x_t[fixed[1].y], fixed[1].x_s)
    none = SPair71kPairs(dataset, 'trn', 'small', TRANSFORM, 'none')
    assert len(none) == 1 and none[0].y.tolist() == [0, 1, 2]
    assert none[0].category == CATEGORIES.index('bicycle')


def test_spair_dgmc_batch(tmp_path):
    dataset = make_dataset(tmp_path)
    pairs = [SPair71kPairs(dataset, split, 'large', TRANSFORM, 'fixed')[i]
             for split in ('trn', 'val', 'test') for i in range(2)]
    data = next(iter(DataLoader(pairs, batch_size=6,
                                follow_batch=['x_s', 'x_t'])))
    assert data.category.tolist() == [0, 1] * 3
    torch.manual_seed(12345)
    psi_1 = SplineCNN(1024, 16, dim=2, num_layers=1, kernel_size=3)
    psi_2 = SplineCNN(8, 8, dim=2, num_layers=1, kernel_size=3)
    model = DGMC(psi_1, psi_2, num_steps=1)
    S_0, S_L = model(data.x_s, data.edge_index_s, data.edge_attr_s,
                     data.x_s_batch, data.x_t, data.edge_index_t,
                     data.edge_attr_t, data.x_t_batch)
    y = torch.stack([torch.arange(data.y.numel()), data.y], dim=0)
    loss = model.loss(S_0, y) + model.loss(S_L, y)
    assert torch.isfinite(loss)
    loss.backward()

    acc = MatchingAccuracy(len(CATEGORIES))
    acc.update(S_L[y[0]].argmax(dim=-1), y[1], data.x_s_batch, data.category)
    out = acc.compute()
    assert out['num_pairs'] == 6
    assert len(out['pair']) == len(CATEGORIES)


def test_spair_max_images(tmp_path):
    dataset = make_dataset(tmp_path, max_images=2)  # keeps a and b only
    assert dataset.processed_dir.endswith('processed_max2')
    assert len(dataset) == 4
    assert dataset.pairs['layout']['large']['trn'].numel() == 2
    assert dataset.pairs['layout']['large']['test'].numel() == 0


def test_matching_accuracy():
    acc = MatchingAccuracy(3)
    # pair 0 (category 0): 1 keypoint, correct; pair 1 (category 0): 1 of 3
    # correct; pair 2 (category 2): 2 of 2 correct. Category 1 has no pairs.
    pred = torch.tensor([0, 0, 2, 0, 0, 1])
    target = torch.tensor([0, 1, 2, 2, 0, 1])
    batch = torch.tensor([0, 1, 1, 1, 2, 2])
    acc.update(pred, target, batch, torch.tensor([0, 0, 2]))
    out = acc.compute()
    assert abs(out['pair'][0] - 100 * (1 + 1 / 3) / 2) < 1e-9
    assert abs(out['kp'][0] - 50.0) < 1e-9
    assert out['pair'][2] == 100.0 and out['kp'][2] == 100.0
    assert abs(out['pair_mean'] - (100 * (1 + 1 / 3) / 2 + 100) / 2) < 1e-9
    assert abs(out['kp_mean'] - 75.0) < 1e-9
    assert out['num_pairs'] == 3
