import torch_geometric.transforms as T


class FaceToEdge(T.FaceToEdge):
    r"""`torch_geometric.transforms.FaceToEdge` that tolerates graphs
    without a `face` attribute (e.g. `T.Delaunay` on fewer than three
    keypoints, which already sets `edge_index`), instead of asserting.
    Restores the pre-PyG-2.4 behaviour the DGMC examples rely on."""
    def forward(self, data):
        if getattr(data, 'face', None) is None:
            return data
        return super().forward(data)
