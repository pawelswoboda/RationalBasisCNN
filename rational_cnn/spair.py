r"""SPair-71k (Min et al., 2019) as a two-graph keypoint matching benchmark for
DGMC, alongside PyG's :class:`PascalVOCKeypoints`.

The protocol follows pygmtools' ``SPair71k``, the deep graph matching benchmark
version of the dataset: the fixed trn/val/test pairs of a layout, no difficulty
filtering, and the ``intersection`` filter -- both graphs of a pair hold exactly
the keypoints visible in both images. That set is what a pair annotation's
``kps_ids`` lists (checked for all 70,958 pairs).

Node features are extracted as in :class:`PascalVOCKeypoints`, so the backbones
see the same input distribution as in the PascalVOC experiment: the bounding box
is grown to contain every visible keypoint plus a 16 px margin (9,404 SPair-71k
keypoints lie outside their box), the crop is resized to 256x256, and the VGG16
``relu4_2`` and ``relu5_1`` maps are bilinearly upsampled and read out at the
rounded keypoint positions (512 + 512 = 1024 channels).
"""
import glob
import json
import os.path as osp
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs

from .data import PairData

CATEGORIES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
              'cat', 'chair', 'cow', 'dog', 'horse', 'motorbike', 'person',
              'pottedplant', 'sheep', 'train', 'tvmonitor']
SPLITS = ('trn', 'val', 'test')
LAYOUTS = ('large', 'small')
DIFFICULTY = ('viewpoint_variation', 'scale_variation', 'truncation',
              'occlusion')
# Pair counts of the released dataset, from SPair-71k/README.
NUM_PAIRS = {'large': {'trn': 53340, 'val': 5384, 'test': 12234},
             'small': {'trn': 10652, 'val': 1070, 'test': 2438}}


def crop_box(pos, bndbox):
    r"""The crop of :class:`PascalVOCKeypoints`: the bounding box grown to
    contain all keypoints, plus a 16 px margin on every side."""
    return (min(int(pos[:, 0].min().floor()), bndbox[0]) - 16,
            min(int(pos[:, 1].min().floor()), bndbox[1]) - 16,
            max(int(pos[:, 0].max().ceil()), bndbox[2]) + 16,
            max(int(pos[:, 1].max().ceil()), bndbox[3]) + 16)


def vgg16_extractor(device):
    r"""Maps a batch of ImageNet-normalised 256x256 images to the VGG16
    ``relu4_2`` and ``relu5_1`` feature maps."""
    import torchvision.models as models

    outputs = []

    def hook(module, x, y):
        outputs.append(y)

    # The weights PascalVOCKeypoints loads through `pretrained=True`.
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    vgg16 = vgg16.to(device).eval()
    vgg16.features[20].register_forward_hook(hook)  # relu4_2
    vgg16.features[25].register_forward_hook(hook)  # relu5_1

    @torch.no_grad()
    def extract(images):
        outputs.clear()
        vgg16(images.to(device))
        return list(outputs)

    return extract


def _locate(kp, ids):
    r"""Node indices of the keypoint ids :obj:`ids` in the sorted :obj:`kp`,
    or :obj:`None` if one of them is missing."""
    idx = torch.searchsorted(kp, ids).clamp(max=max(kp.numel() - 1, 0))
    return idx if bool((kp[idx] == ids).all()) else None


class SPair71k(InMemoryDataset):
    r"""The 1,800 SPair-71k images as keypoint graphs over all visible
    keypoints (VGG16 features :obj:`x`, positions :obj:`pos` in the 256x256
    crop, keypoint ids :obj:`kp`), with the pair tables of both layouts in
    :obj:`self.pairs`. Build graph pairs with :class:`SPair71kPairs`.

    Args:
        root (str): Directory holding the processed files.
        raw_dir (str, optional): The extracted SPair-71k tree. Only read when
            the processed files do not exist yet. (default: :obj:`None`)
        max_images (int, optional): For debugging, keep only the first
            :obj:`max_images` images of every category and the pairs among
            them, stored separately in :obj:`processed_max<N>/`.
            (default: :obj:`None`)
        device (str, optional): Device of the VGG16 forward passes.
            (default: CUDA if available)
        batch_size (int): Images per VGG16 forward pass. (default: :obj:`32`)
        extractor (callable, optional): Replaces VGG16, e.g. in tests. Maps a
            :obj:`[B, 3, 256, 256]` batch to a list of two feature maps.
        force_reload (bool): Re-process even if the processed files exist.
    """
    categories = CATEGORIES

    def __init__(self, root: str, raw_dir: Optional[str] = None,
                 max_images: Optional[int] = None,
                 device: Optional[str] = None, batch_size: int = 32,
                 extractor: Optional[Callable] = None,
                 force_reload: bool = False) -> None:
        self._raw_dir = raw_dir
        self.max_images = max_images
        self.device = device or ('cuda' if torch.cuda.is_available() else
                                 'cpu')
        self.batch_size = batch_size
        self.extractor = extractor
        super().__init__(root, force_reload=force_reload)
        self.load(self.processed_paths[0])
        self.pairs = fs.torch_load(self.processed_paths[1])

    @property
    def raw_dir(self) -> str:
        return self._raw_dir or osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        name = ('processed' if self.max_images is None else
                f'processed_max{self.max_images}')
        return osp.join(self.root, name)

    @property
    def raw_file_names(self) -> List[str]:
        return ['ImageAnnotation', 'JPEGImages', 'Layout', 'PairAnnotation']

    @property
    def processed_file_names(self) -> List[str]:
        return ['images.pt', 'pairs.pt']

    # There is deliberately no `download()`: PyG then never checks the raw
    # files, so processed files work without the raw tree and on a read-only
    # filesystem (group storage inside Slurm jobs). Only `process()` needs it.

    def process(self) -> None:
        missing = [p for p in self.raw_paths if not osp.exists(p)]
        if missing:
            raise FileNotFoundError(
                f'SPair-71k raw data not found ({missing[0]}). Download it '
                f'with hpc/download_spair71k.sh and pass the extracted tree '
                f'as raw_dir.')
        images, index = self._process_images()
        pairs = self._process_pairs(images, index)
        self.save(images, self.processed_paths[0])
        fs.torch_save(pairs, self.processed_paths[1])

    def _process_images(self):
        import torchvision.transforms as T
        from PIL import Image

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        images, index = [], {}
        for c, category in enumerate(self.categories):
            paths = sorted(glob.glob(osp.join(self.raw_dir, 'ImageAnnotation',
                                              category, '*.json')))
            if self.max_images is not None:
                paths = paths[:self.max_images]
            for path in paths:
                with open(path) as f:
                    ann = json.load(f)
                kps = sorted((int(k), v) for k, v in ann['kps'].items()
                             if v is not None)
                if len(kps) == 0:
                    continue  # cannot be part of any pair
                kp = torch.tensor([k for k, _ in kps], dtype=torch.long)
                pos = torch.tensor([v for _, v in kps], dtype=torch.float)

                box = crop_box(pos, ann['bndbox'])
                pos[:, 0] = (pos[:, 0] - box[0]) * 256.0 / (box[2] - box[0])
                pos[:, 1] = (pos[:, 1] - box[1]) * 256.0 / (box[3] - box[1])

                path = osp.join(self.raw_dir, 'JPEGImages', category,
                                ann['filename'])
                with open(path, 'rb') as f:
                    img = Image.open(f).convert('RGB').crop(box)
                    img = img.resize((256, 256),
                                     resample=Image.Resampling.BICUBIC)

                name = osp.splitext(ann['filename'])[0]
                index[category, name] = len(images)
                images.append(Data(img=transform(img), pos=pos, kp=kp,
                                   category=c, name=name))

        extract = self.extractor or vgg16_extractor(self.device)
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            maps = extract(torch.stack([data.img for data in batch]))
            out1 = F.interpolate(maps[0], (256, 256), mode='bilinear',
                                 align_corners=False)
            out2 = F.interpolate(maps[1], (256, 256), mode='bilinear',
                                 align_corners=False)
            for j, data in enumerate(batch):
                idx = data.pos.round().long().clamp(0, 255)
                x_1 = out1[j, :, idx[:, 1], idx[:, 0]].to('cpu')
                x_2 = out2[j, :, idx[:, 1], idx[:, 0]].to('cpu')
                data.x = torch.cat([x_1.t(), x_2.t()], dim=-1)
                del data.img
            del out1, out2

        return images, index

    def _process_pairs(self, images, index):
        kp = [data.kp for data in images]
        src, trg, category, split_of, names = [], [], [], [], []
        src_nodes, trg_nodes, ptr = [], [], [0]
        difficulty = {key: [] for key in DIFFICULTY}
        row = {}

        for s, split in enumerate(SPLITS):
            paths = sorted(glob.glob(osp.join(self.raw_dir, 'PairAnnotation',
                                              split, '*.json')))
            for path in paths:
                with open(path) as f:
                    ann = json.load(f)
                cat = ann['category']
                i = index.get((cat, osp.splitext(ann['src_imname'])[0]))
                j = index.get((cat, osp.splitext(ann['trg_imname'])[0]))
                if i is None or j is None:
                    if self.max_images is None:
                        raise ValueError(f'{path}: unknown image')
                    continue

                ids = torch.tensor(sorted(int(k) for k in ann['kps_ids']),
                                   dtype=torch.long)
                u, v = _locate(kp[i], ids), _locate(kp[j], ids)
                if u is None or v is None:
                    raise ValueError(f'{path}: kps_ids not visible in both '
                                     f'images')

                name = osp.splitext(osp.basename(path))[0]
                row[split, name] = len(src)
                src.append(i)
                trg.append(j)
                category.append(self.categories.index(cat))
                split_of.append(s)
                names.append(name)
                src_nodes.append(u)
                trg_nodes.append(v)
                ptr.append(ptr[-1] + ids.numel())
                for key in DIFFICULTY:
                    difficulty[key].append(int(ann[key]))

        layouts = {}
        for layout in LAYOUTS:
            layouts[layout] = {}
            for split in SPLITS:
                path = osp.join(self.raw_dir, 'Layout', layout, f'{split}.txt')
                with open(path) as f:
                    ids = [line.strip() for line in f if line.strip()]
                rows = [row[split, i] for i in ids if (split, i) in row]
                if self.max_images is None and len(rows) != len(ids):
                    raise ValueError(f'{path}: {len(ids) - len(rows)} pairs '
                                     f'without an annotation')
                layouts[layout][split] = torch.tensor(rows, dtype=torch.long)

        def long(xs):
            return torch.tensor(xs, dtype=torch.long)

        return {
            'src': long(src),
            'trg': long(trg),
            'category': long(category),
            'split': long(split_of),
            'name': names,
            'ptr': long(ptr),
            'src_nodes': torch.cat(src_nodes) if src_nodes else long([]),
            'trg_nodes': torch.cat(trg_nodes) if trg_nodes else long([]),
            'difficulty': {k: long(v) for k, v in difficulty.items()},
            'layout': layouts,
        }

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)})'


class SPair71kPairs(torch.utils.data.Dataset):
    r"""The graph pairs of one split and layout of a :class:`SPair71k`.

    Every item is a :class:`PairData` whose source and target graphs hold the
    keypoints visible in both images, with :obj:`y` mapping each source node
    to its target node and the pair's :obj:`category` index. Graphs are built
    on the fly by :obj:`transform` (Delaunay triangulation and pseudo-
    coordinates), as in the PascalVOC experiment.

    The target nodes are permuted: the annotated keypoint order is the same in
    both images, so the unpermuted ground truth is the identity, and argmax
    ties (which resolve to the first index) would be scored as correct.

    Args:
        dataset (SPair71k): The processed images and pair tables.
        split (str): :obj:`"trn"`, :obj:`"val"` or :obj:`"test"`.
        layout (str): :obj:`"large"` (the benchmark) or :obj:`"small"`.
            (default: :obj:`"large"`)
        transform (callable, optional): Applied to every graph.
        shuffle (str): :obj:`"random"` draws a new target permutation on every
            access (training), :obj:`"fixed"` uses one permutation per pair
            that is identical in every run (evaluation), :obj:`"none"` keeps
            the annotated order. (default: :obj:`"random"`)
    """
    def __init__(self, dataset: SPair71k, split: str, layout: str = 'large',
                 transform: Optional[Callable] = None,
                 shuffle: str = 'random') -> None:
        assert split in SPLITS and layout in LAYOUTS
        assert shuffle in ('random', 'fixed', 'none')
        pairs = dataset.pairs
        self.split, self.layout = split, layout
        self.transform, self.shuffle = transform, shuffle
        self.index = pairs['layout'][layout][split]
        self.src, self.trg = pairs['src'], pairs['trg']
        self.category, self.ptr = pairs['category'], pairs['ptr']
        self.src_nodes, self.trg_nodes = pairs['src_nodes'], pairs['trg_nodes']
        self.x = [data.x for data in dataset]
        self.pos = [data.pos for data in dataset]

    def __len__(self) -> int:
        return self.index.numel()

    def graph(self, image: int, nodes: torch.Tensor) -> Data:
        data = Data(x=self.x[image][nodes], pos=self.pos[image][nodes])
        if self.transform is not None:
            data = self.transform(data)
        if data.edge_attr is not None:
            # T.Cartesian / T.Distance normalise by the largest offset, which
            # is 0/0 when all nodes coincide. No SPair-71k graph does; this
            # only keeps one degenerate graph from turning a batch into NaN.
            data.edge_attr = torch.nan_to_num(data.edge_attr, nan=0.5)
        return data

    def __getitem__(self, idx: int) -> PairData:
        p = int(self.index[idx])
        lo, hi = int(self.ptr[p]), int(self.ptr[p + 1])
        n = hi - lo
        if self.shuffle == 'random':
            perm = torch.randperm(n)
        elif self.shuffle == 'fixed':
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(p))
        else:
            perm = torch.arange(n)

        # Target node k is the keypoint of source node perm[k].
        y = torch.empty(n, dtype=torch.long)
        y[perm] = torch.arange(n)

        s = self.graph(int(self.src[p]), self.src_nodes[lo:hi])
        t = self.graph(int(self.trg[p]), self.trg_nodes[lo:hi][perm])
        return PairData(
            x_s=s.x,
            edge_index_s=s.edge_index,
            edge_attr_s=s.edge_attr,
            x_t=t.x,
            edge_index_t=t.edge_index,
            edge_attr_t=t.edge_attr,
            y=y,
            category=int(self.category[p]),
            num_nodes=None,
        )

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, split={self.split}, '
                f'layout={self.layout}, shuffle={self.shuffle})')


class MatchingAccuracy:
    r"""Per-category matching accuracy over graph pairs, both

    * pair-averaged: the mean over pairs of the fraction of correctly matched
      source keypoints (the pygmtools benchmark convention, and the headline
      number), and
    * keypoint-weighted: correct keypoints over all keypoints (the convention
      of experiments/pascal_voc.py, which down-weights few-keypoint pairs --
      1,860 SPair-71k pairs have a single keypoint and are always correct).

    Category means average over the categories that have at least one pair.
    """
    def __init__(self, num_categories: int) -> None:
        zeros = lambda: torch.zeros(num_categories, dtype=torch.float64)  # noqa
        self.pair_acc, self.pairs = zeros(), zeros()
        self.correct, self.keypoints = zeros(), zeros()

    @torch.no_grad()
    def update(self, pred, target, batch, category) -> None:
        r"""
        Args:
            pred (LongTensor): Predicted target node per source node.
            target (LongTensor): Ground-truth target node per source node.
            batch (LongTensor): Pair index of every source node.
            category (LongTensor): Category of every pair.
        """
        B = category.numel()
        correct = (pred == target).to(torch.float64)
        c = correct.new_zeros(B).scatter_add_(0, batch, correct).cpu()
        n = torch.bincount(batch, minlength=B).to(torch.float64).cpu()
        category = category.cpu()
        self.pair_acc.index_add_(0, category, c / n)
        self.pairs.index_add_(0, category, torch.ones_like(n))
        self.correct.index_add_(0, category, c)
        self.keypoints.index_add_(0, category, n)

    def compute(self) -> dict:
        valid = self.pairs > 0
        pair = 100 * self.pair_acc / self.pairs.clamp(min=1)
        kp = 100 * self.correct / self.keypoints.clamp(min=1)
        return {
            'pair': pair.tolist(),
            'kp': kp.tolist(),
            'pair_mean': pair[valid].mean().item(),
            'kp_mean': kp[valid].mean().item(),
            'num_pairs': int(self.pairs.sum()),
        }
