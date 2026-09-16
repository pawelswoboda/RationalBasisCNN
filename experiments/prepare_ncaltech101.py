r"""Pre-process N-Caltech101 for the AEGNN recognition experiment
(experiments/ncaltech101.py), following the released AEGNN pipeline
(aegnn/datasets/ncaltech101.py, Schaefer et al., CVPR 2022):

1. Read the event files (x, y, t, polarity); the archive of Gehrig et al.
   stores each recording as a float32 ``.npy`` array with t in seconds
   (raw ATIS ``.bin`` files are supported too).
2. Keep the 50 ms window of events that ends at the median event.
3. Sample exactly ``n_samples`` = 25000 events uniformly without replacement
   (cycling through permutations if fewer events are available, like PyG's
   ``FixedPoints(allow_duplicates=False)``).
4. Rescale time by ``beta`` = 0.5e-5 per microsecond so that it is
   comparable to the pixel coordinates.

The radius graph (r = 5, at most 32 neighbours) is *not* stored; it is built
on the GPU for every batch by the training script, which is equivalent to
AEGNN's offline graph but keeps the processed data at ~2.7 GB.

Note: the released AEGNN code converts the timestamps to seconds before
applying the 50 ms window and ``beta`` (which are in microseconds), so that in
their pre-processing the window spans everything up to the median event and
the temporal coordinate collapses to ~0. We apply both in microseconds, as
described in the paper.

Data: the training / validation / test split of Gehrig et al. (ICCV 2019),
https://download.ifi.uzh.ch/rpg/web/datasets/gehrig_et_al_iccv19/N-Caltech101.zip
(N-Caltech101 by Orchard et al., CC BY 4.0). Pass ``--download`` to fetch it.
"""
import argparse
import json
import os
import os.path as osp
import sys
import zipfile

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from event_utils import class_files, write_split  # noqa: E402

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
URL = ('https://download.ifi.uzh.ch/rpg/web/datasets/gehrig_et_al_iccv19/'
       'N-Caltech101.zip')

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default=osp.join(ROOT, 'data'))
parser.add_argument('--download', action='store_true')
parser.add_argument('--n_samples', type=int, default=25000)
parser.add_argument('--window_us', type=float, default=50 * 1000)
parser.add_argument('--beta', type=float, default=0.5e-5)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=0)


def find_split_dir(raw_dir, name):
    alt = {'test': 'testing', 'testing': 'test'}.get(name, name)
    for cand in [name, alt]:
        for d in [osp.join(raw_dir, 'N-Caltech101', cand),
                  osp.join(raw_dir, cand)]:
            if osp.isdir(d):
                return d
    raise FileNotFoundError(f'split "{name}" not found under {raw_dir}')


if __name__ == '__main__':
    args = parser.parse_args()
    path = osp.join(args.root, 'NCaltech101')
    raw_dir, processed_dir = osp.join(path, 'raw'), osp.join(path, 'processed')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    zip_path = osp.join(raw_dir, 'N-Caltech101.zip')

    if not osp.exists(zip_path) and args.download:
        import urllib.request
        print(f'Downloading {URL} (5.9 GB) ...')
        urllib.request.urlretrieve(URL, zip_path)
    if not osp.isdir(osp.join(raw_dir, 'N-Caltech101')) and \
            not osp.isdir(osp.join(raw_dir, 'training')):
        assert osp.exists(zip_path), f'{zip_path} missing (use --download)'
        print(f'Extracting {zip_path} ...')
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)

    train_dir = find_split_dir(raw_dir, 'training')
    classes = sorted(d for d in os.listdir(train_dir)
                     if osp.isdir(osp.join(train_dir, d)))
    print(f'{len(classes)} classes')
    with open(osp.join(processed_dir, 'classes.json'), 'w') as f:
        json.dump(classes, f)
    for split in ['training', 'validation', 'test']:
        files, labels = class_files(find_split_dir(raw_dir, split), classes,
                                    ('.bin', '.npy'))
        write_split(processed_dir, split, files, labels, args.n_samples,
                    args.beta, window_us=args.window_us,
                    workers=args.workers, seed=args.seed, rel_to=raw_dir)
    print('done')
