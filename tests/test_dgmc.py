import torch
from torch_geometric.data import Data, Batch

from rational_cnn import DGMC, SplineCNN, RationalCNN

x = torch.randn(4, 32)
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
edge_attr = torch.rand(edge_index.size(1), 2)
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def set_seed():
    torch.manual_seed(12345)


def make_model(backbone):
    psi_1 = backbone(data.num_node_features, 16, dim=2, num_layers=2,
                     kernel_size=3)
    psi_2 = backbone(8, 8, dim=2, num_layers=2, kernel_size=3)
    return DGMC(psi_1, psi_2, num_steps=1)


def test_dgmc_repr():
    model = make_model(SplineCNN)
    assert model.__repr__() == (
        'DGMC(\n'
        '    psi_1=SplineCNN(32, 16, dim=2, num_layers=2, cat=True, '
        'lin=True, dropout=0.0),\n'
        '    psi_2=SplineCNN(8, 8, dim=2, num_layers=2, cat=True, '
        'lin=True, dropout=0.0),\n'
        '    num_steps=1, k=-1\n)')
    model.reset_parameters()


def test_dgmc_dense_vs_sparse():
    for backbone in (SplineCNN, RationalCNN):
        set_seed()
        model = make_model(backbone)
        x, e, a = data.x, data.edge_index, data.edge_attr
        y = torch.arange(data.num_nodes)
        y = torch.stack([y, y], dim=0)

        set_seed()
        S1_0, S1_L = model(x, e, a, None, x, e, a, None)
        loss1 = model.loss(S1_0, y)
        loss1.backward()
        acc1 = model.acc(S1_0, y)
        hits1_all = model.hits_at_k(data.num_nodes, S1_0, y)

        set_seed()
        model.k = data.num_nodes  # Sparse "dense" variant.
        S2_0, S2_L = model(x, e, a, None, x, e, a, None, y)
        loss2 = model.loss(S2_0, y)
        acc2 = model.acc(S2_0, y)

        assert S1_0.size() == (data.num_nodes, data.num_nodes)
        assert S1_L.size() == (data.num_nodes, data.num_nodes)
        assert torch.allclose(S1_0, S2_0.to_dense(), atol=1e-6)
        assert torch.allclose(S1_L, S2_L.to_dense(), atol=1e-6)
        assert torch.allclose(loss1, loss2)
        assert acc1 == acc2
        assert hits1_all == 1.0


def test_dgmc_on_multiple_graphs():
    set_seed()
    model = make_model(SplineCNN)
    batch = Batch.from_data_list([data, data])
    x, e, a, b = batch.x, batch.edge_index, batch.edge_attr, batch.batch

    set_seed()
    S1_0, S1_L = model(x, e, a, b, x, e, a, b)
    assert S1_0.size() == (batch.num_nodes, data.num_nodes)
    assert S1_L.size() == (batch.num_nodes, data.num_nodes)

    set_seed()
    model.k = data.num_nodes
    S2_0, S2_L = model(x, e, a, b, x, e, a, b)
    assert torch.allclose(S1_0, S2_0.to_dense(), atol=1e-6)
    assert torch.allclose(S1_L, S2_L.to_dense(), atol=1e-6)


def test_dgmc_include_gt():
    model = make_model(SplineCNN)
    S_idx = torch.tensor([[[0, 1], [1, 2]], [[1, 2], [0, 1]]])
    s_mask = torch.tensor([[True, False], [True, True]])
    y = torch.tensor([[0, 1], [0, 0]])
    S_idx = model.__include_gt__(S_idx, s_mask, y)
    assert S_idx.tolist() == [[[0, 1], [1, 2]], [[1, 0], [0, 1]]]
