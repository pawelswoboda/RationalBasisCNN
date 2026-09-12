import numpy as np
import torch
from torch_geometric.data import Data


def read_ply(path):
    r"""`plyfile`-based replacement for PyG's `torch_geometric.io.read_ply`
    (which needs the `openmesh` package, for which no recent wheels exist).
    Assign it to `torch_geometric.datasets.faust.read_ply` before the FAUST
    dataset is processed."""
    from plyfile import PlyData
    ply = PlyData.read(path)
    v = ply['vertex']
    pos = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
    f = ply['face']
    key = 'vertex_indices' if 'vertex_indices' in f.data.dtype.names \
        else 'vertex_index'
    face = np.stack(f[key]).astype(np.int64)
    return Data(pos=torch.from_numpy(pos),
                face=torch.from_numpy(face).t().contiguous())
