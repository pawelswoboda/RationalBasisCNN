r"""Places the FAUST registrations where PyG's `FAUST` dataset expects them
and processes the dataset once (using the `plyfile` reader, since PyG's needs
`openmesh`). FAUST must be obtained from http://faust.is.tue.mpg.de/ (its
license forbids redistribution). Usage:

    python experiments/prepare_faust.py --src /path/to/MPI-FAUST.zip
    python experiments/prepare_faust.py --src /path/to/MPI-FAUST/   # unzipped

The zip must contain `MPI-FAUST/training/registrations/tr_reg_*.ply`.
"""
import argparse
import os
import os.path as osp
import shutil
import sys
import zipfile

import torch_geometric.transforms as T
import torch_geometric.datasets.faust as faust_module
from torch_geometric.datasets import FAUST

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
# Redirected to project storage by hpc/env.sh; $HOME is quota-limited.
DEFAULT_ROOT = os.environ.get('RBCNN_DATA_ROOT', osp.join(ROOT, 'data'))
from rational_cnn import FaceToEdge, read_ply  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, required=True,
                    help='MPI-FAUST.zip or the unzipped MPI-FAUST directory')
parser.add_argument('--root', type=str, default=DEFAULT_ROOT)
args = parser.parse_args()

path = osp.join(args.root, 'FAUST')
raw_dir = osp.join(path, 'raw')
os.makedirs(raw_dir, exist_ok=True)
dst = osp.join(raw_dir, 'MPI-FAUST.zip')
if not osp.exists(dst):
    if osp.isdir(args.src):
        reg = osp.join(args.src, 'training', 'registrations')
        plys = sorted(f for f in os.listdir(reg) if f.endswith('.ply'))
        assert len(plys) == 100, f'expected 100 registrations in {reg}'
        with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as z:
            for f in plys:
                z.write(osp.join(reg, f),
                        f'MPI-FAUST/training/registrations/{f}')
    else:
        shutil.copyfile(args.src, dst)
    print(f'wrote {dst}')

faust_module.read_ply = read_ply
pre_transform = T.Compose([FaceToEdge(), T.Constant(value=1)])
train = FAUST(path, True, T.Cartesian(), pre_transform)
test = FAUST(path, False, T.Cartesian(), pre_transform)
print(f'FAUST: {len(train)} train / {len(test)} test meshes, '
      f'{train[0].num_nodes} vertices, {train[0].num_edges} edges')
