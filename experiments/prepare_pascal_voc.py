r"""Downloads PascalVOC-Keypoints once and runs the VGG16 feature extraction
for all 20 categories, so that training jobs do not race on it.

PyG's `PascalVOCKeypoints.download()` fetches the keypoint annotations from a
Berkeley URL that is dead; this script mirrors that download step but takes
the annotations from the Internet Archive instead (images and splits come
from PyG's original sources). Usage:

    python experiments/prepare_pascal_voc.py [--root data]
"""
import argparse
import os
import os.path as osp
import sys
import time

from torch_geometric.data import download_url, extract_tar
from torch_geometric.datasets import PascalVOCKeypoints as PascalVOC
from torch_geometric.io import fs

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)

ANNOTATION_URL = ('https://web.archive.org/web/20240416142946id_/'
                  'https://www2.eecs.berkeley.edu/Research/Projects/CS/'
                  'vision/shape/poselets/voc2011_keypoints_Feb2012.tgz')

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default=osp.join(ROOT, 'data'))
args = parser.parse_args()

path = osp.join(args.root, 'PascalVOC')
raw_dir = osp.join(path, 'raw')
os.makedirs(raw_dir, exist_ok=True)

if not osp.isdir(osp.join(raw_dir, 'images')):
    p = download_url(PascalVOC.image_url, raw_dir)
    extract_tar(p, raw_dir, mode='r')
    os.unlink(p)
    os.rename(osp.join(raw_dir, 'TrainVal', 'VOCdevkit', 'VOC2011'),
              osp.join(raw_dir, 'images'))
    fs.rm(osp.join(raw_dir, 'TrainVal'))
if not osp.isdir(osp.join(raw_dir, 'annotations')):
    p = download_url(ANNOTATION_URL, raw_dir,
                     filename='voc2011_keypoints_Feb2012.tgz')
    extract_tar(p, raw_dir, mode='r')
    os.unlink(p)
if not osp.exists(osp.join(raw_dir, 'splits.npz')):
    p = download_url(PascalVOC.split_url, raw_dir)
    os.rename(p, osp.join(raw_dir, 'splits.npz'))

t = time.time()
pre_filter = lambda data: data.pos.size(0) > 0  # noqa  (same as pascal_voc.py)
for category in PascalVOC.categories:
    for train in (True, False):
        ds = PascalVOC(path, category, train=train, pre_filter=pre_filter)
        print(f'{category:12s} train={train!s:5s} {len(ds):5d} graphs',
              flush=True)
print(f'done in {time.time() - t:.0f}s')
