r"""Shared pre-processing for the event-camera experiments (AEGNN pipeline,
Schaefer et al., CVPR 2022): event file readers, the window / fixed-size
subsampling / time rescaling of ``NCaltech101.pre_transform`` and
``NCars.pre_transform``, and writers for the memory-mapped split files read by
``experiments/ncaltech101.py``."""
import json
import math
import os
import os.path as osp
from multiprocessing import Pool

import numpy as np


def load_npy_events(raw_file):
    r"""Gehrig et al. (ICCV 2019) N-Caltech101 archive: float32 (x, y, t [s],
    p) -> float64 (x, y, t [us], p)."""
    ev = np.load(raw_file).astype(np.float64)
    ev[:, 2] *= 1e6
    return ev


def load_atis_bin(raw_file):
    r"""ATIS binary format (Orchard et al.), as in AEGNN's
    ``NCaltech101.load``: returns float64 (x, y, t [us], p in {-1, 1})."""
    raw = np.fromfile(raw_file, dtype=np.uint8).astype(np.uint32)
    x = raw[0::5]
    y = raw[1::5]
    p = (raw[2::5] & 128) >> 7
    t = ((raw[2::5] & 127) << 16) | (raw[3::5] << 8) | raw[4::5]
    p = p.astype(np.float64)
    p[p == 0] = -1
    return np.column_stack((x, y, t, p)).astype(np.float64)


def load_prophesee_dat(raw_file):
    r"""Prophesee ``_td.dat`` (N-Cars): '%'-prefixed ASCII header, 2 header
    bytes (event type / size), then uint32 pairs (t [us], packed data with
    14-bit x, 14-bit y, 1-bit polarity). Returns float64 (x, y, t, p)."""
    with open(raw_file, 'rb') as f:
        pos = 0
        while True:
            line = f.readline()
            if not line.startswith(b'%'):
                break
            pos = f.tell()
        f.seek(pos)
        f.read(2)  # event type, event size
        raw = np.fromfile(f, dtype=np.uint32)
    t = raw[0::2].astype(np.float64)
    d = raw[1::2]
    x = (d & 0x3FFF).astype(np.float64)
    y = ((d & 0xFFFC000) >> 14).astype(np.float64)
    p = ((d & 0x10000000) >> 28).astype(np.float64)
    p[p == 0] = -1
    return np.column_stack((x, y, t, p))


def load_txt_events(raw_file):
    r"""AEGNN's converted N-Cars ``events.txt`` (x y t p per line)."""
    return np.loadtxt(raw_file).astype(np.float64)


def load_events(raw_file):
    if raw_file.endswith('.npy'):
        return load_npy_events(raw_file)
    if raw_file.endswith('.bin'):
        return load_atis_bin(raw_file)
    if raw_file.endswith('.dat'):
        return load_prophesee_dat(raw_file)
    if raw_file.endswith('.txt'):
        return load_txt_events(raw_file)
    raise ValueError(raw_file)


def window_median(events, window_us):
    r"""AEGNN ``NCaltech101.pre_transform``: the ``window_us`` window of
    events ending at the median event (timestamps sorted)."""
    t = events[:, 2]
    n = len(t)
    t_mid = t[n // 2]
    i1 = int(np.clip(np.searchsorted(t, t_mid) - 1, 0, n - 1))
    i0 = int(np.clip(np.searchsorted(t, t_mid - window_us) - 1, 0, n - 1))
    return events[i0:i1] if i1 > i0 else events


def fixed_points(events, n_samples, rng):
    r"""PyG ``FixedPoints(n_samples, allow_duplicates=False,
    replace=False)``: a random subset of exactly ``n_samples`` events
    (cycling through permutations if there are fewer)."""
    n = len(events)
    perm = np.concatenate([rng.permutation(n)
                           for _ in range(math.ceil(n_samples / n))])
    return events[perm[:n_samples]]


def pre_transform(events, n_samples, beta, rng, window_us=None):
    r"""Returns float32 ``pos`` (x, y, beta * (t - t_min)) of shape
    ``[n_samples, 3]`` and int8 polarities."""
    if window_us:
        events = window_median(events, window_us)
    events = fixed_points(events, n_samples, rng)
    pos = events[:, :3].copy()
    pos[:, 2] = (pos[:, 2] - pos[:, 2].min()) * beta
    return pos.astype(np.float32), events[:, 3].astype(np.int8)


_CFG = {}


def _process_one(task):
    i, raw_file, seed = task
    rng = np.random.default_rng(seed)
    pos, x = pre_transform(load_events(raw_file), _CFG['n_samples'],
                           _CFG['beta'], rng, _CFG.get('window_us'))
    return i, pos, x


def write_split(processed_dir, split, files, labels, n_samples, beta,
                window_us=None, workers=4, seed=0, rel_to=None):
    r"""Pre-processes ``files`` (with integer ``labels``) into
    ``<split>_pos.npy`` [S, n_samples, 3] float32, ``<split>_x.npy``
    [S, n_samples] int8, ``<split>_y.npy`` [S] and ``<split>_files.json``."""
    _CFG.update(n_samples=n_samples, beta=beta, window_us=window_us)
    S = len(files)
    print(f'{split}: {S} samples', flush=True)
    pos = np.lib.format.open_memmap(
        osp.join(processed_dir, f'{split}_pos.npy'), mode='w+',
        dtype=np.float32, shape=(S, n_samples, 3))
    x = np.lib.format.open_memmap(
        osp.join(processed_dir, f'{split}_x.npy'), mode='w+', dtype=np.int8,
        shape=(S, n_samples))
    base = {'training': 0, 'validation': 1, 'test': 2}.get(split, 3)
    tasks = [(i, f, seed * 10**7 + base * 10**4 + i)
             for i, f in enumerate(files)]
    with Pool(workers) as pool:
        for k, (i, p, xi) in enumerate(pool.imap_unordered(_process_one,
                                                           tasks, 8)):
            pos[i], x[i] = p, xi
            if k % 500 == 0:
                print(f'  {k}/{S}', flush=True)
    pos.flush()
    x.flush()
    np.save(osp.join(processed_dir, f'{split}_y.npy'),
            np.array(labels, dtype=np.int64))
    with open(osp.join(processed_dir, f'{split}_files.json'), 'w') as f:
        json.dump([osp.relpath(fn, rel_to) if rel_to else fn
                   for fn in files], f)


def class_files(split_dir, classes, exts):
    files, labels = [], []
    for c, cls in enumerate(classes):
        for fn in sorted(os.listdir(osp.join(split_dir, cls))):
            if fn.endswith(exts):
                files.append(osp.join(split_dir, cls, fn))
                labels.append(c)
    return files, labels
