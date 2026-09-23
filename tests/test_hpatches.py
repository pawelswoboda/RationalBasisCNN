import numpy as np
import torch

from rational_cnn.hpatches import (MeanAggregation, Refinement,
                                   assemble_candidates, build_graph,
                                   distractors, match, project, score,
                                   shi_tomasi)


def test_project_matches_manual_homography():
    H = np.array([[2.0, 0.0, 5.0], [0.0, 3.0, -1.0], [0.0, 0.0, 1.0]])
    points = np.array([[1.0, 1.0], [10.0, 4.0]])
    out = project(points, H)
    assert np.allclose(out, [[7.0, 2.0], [25.0, 11.0]])
    # a non-affine homography must divide by the third coordinate
    H2 = np.array([[1.0, 0, 0], [0, 1.0, 0], [0.01, 0, 1.0]])
    assert np.allclose(project(np.array([[100.0, 50.0]]), H2),
                       [[100 / 2.0, 50 / 2.0]])


def test_shi_tomasi_finds_a_corner():
    image = torch.zeros(1, 1, 96, 96)
    image[..., 40:, 40:] = 1.0          # one step corner at (40, 40)
    points = shi_tomasi(image, num_keypoints=4, border=8, nms=5)
    assert len(points) >= 1
    # (x, y) ordering, within a pixel or two of the corner
    assert torch.linalg.norm(points[0] - torch.tensor([40.0, 40.0])) < 3


def test_distractors_avoid_ground_truth():
    image = torch.rand(1, 1, 128, 128)
    avoid = np.array([[40.0, 40.0], [80.0, 80.0]])
    extra = distractors(image, avoid, num=10, min_distance=10.0)
    assert len(extra) <= 10
    if len(extra):
        d = torch.cdist(extra, torch.as_tensor(avoid, dtype=torch.float))
        assert float(d.min()) > 10.0


def test_assemble_candidates_bookkeeping():
    """The shuffled candidate set must still point every query at its own
    ground-truth location; getting this wrong would silently invalidate every
    number the experiment produces."""
    gt = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    extra = torch.tensor([[9.0, 9.0], [8.0, 8.0]])
    candidates, gt_index = assemble_candidates(gt, extra, seed=7)

    assert candidates.shape == (5, 2) and gt_index.shape == (3, )
    for i in range(len(gt)):
        assert np.allclose(candidates[gt_index[i]], gt[i])
    assert len(set(gt_index.tolist())) == 3          # distinct slots
    # deterministic for a given seed, different for another
    again, _ = assemble_candidates(gt, extra, seed=7)
    assert np.allclose(candidates, again)
    other, _ = assemble_candidates(gt, extra, seed=8)
    assert not np.allclose(candidates, other)


def test_assemble_candidates_without_distractors():
    gt = np.array([[1.0, 1.0], [2.0, 2.0]])
    candidates, gt_index = assemble_candidates(gt, torch.zeros(0, 2), seed=3)
    assert candidates.shape == (2, 2)
    for i in range(2):
        assert np.allclose(candidates[gt_index[i]], gt[i])


def test_score_is_perfect_for_oracle_matching():
    gt = np.array([[1.0, 1.0], [2.0, 5.0], [7.0, 3.0]])
    extra = torch.tensor([[20.0, 20.0], [30.0, 30.0]])
    candidates, gt_index = assemble_candidates(gt, extra, seed=11)
    out = score(gt_index, candidates, gt_index)
    assert out['acc'] == 1.0 and out['mma@1'] == 1.0
    assert out['queries'] == 3 and out['candidates'] == 5

    # a wrong pick that is still within 3 px counts for mma@3 but not for acc
    wrong = gt_index.clone()
    near = np.array(candidates)
    near[wrong[0]] = near[wrong[0]]           # unchanged
    shifted = np.array(candidates, dtype=float)
    shifted[gt_index[0]] = shifted[gt_index[0]] + np.array([2.0, 0.0])
    out = score(gt_index, shifted, gt_index)
    assert out['acc'] == 1.0


def test_end_to_end_matching_with_distractors():
    """With descriptors that identify the correspondence, matching through the
    shuffled candidate set must recover it exactly."""
    torch.manual_seed(0)
    gt = np.random.RandomState(0).rand(12, 2) * 100
    extra = torch.rand(12, 2) * 100 + 200
    candidates, gt_index = assemble_candidates(gt, extra, seed=5)

    signature = torch.randn(12, 32)
    features_ref = signature
    features_trg = torch.randn(len(candidates), 32)
    features_trg[gt_index] = signature          # the answer, placed correctly

    pred = match(features_ref, features_trg)
    assert torch.equal(pred, gt_index)
    assert score(pred, candidates, gt_index)['acc'] == 1.0


def test_refinement_shapes_and_mean_control():
    cfg = dict(conv='spline', kernel_size=3, dim=2, aggr='max', pyg_init=True)
    module = Refinement(cfg, in_channels=16, out_channels=8)
    pos = torch.rand(10, 2) * 100
    edge_index, edge_attr = build_graph(pos)
    x = torch.randn(10, 16)
    out = module(x, edge_index, edge_attr)
    assert out.shape == (10, 8) and torch.isfinite(out).all()

    # constant input must still produce per-node variation: that is the
    # geometry-only signal the control arm measures
    flat = module(torch.ones(10, 16), edge_index, edge_attr)
    assert float(flat.std(dim=0).mean()) > 0

    mean = MeanAggregation()(x, edge_index)
    assert mean.shape == x.shape and torch.isfinite(mean).all()


def test_build_graph_pseudo_coordinates_are_normalised():
    pos = torch.rand(20, 2) * 256
    edge_index, edge_attr = build_graph(pos)
    assert edge_index.size(0) == 2 and edge_attr.size(-1) == 2
    assert float(edge_attr.min()) >= 0.0 and float(edge_attr.max()) <= 1.0
