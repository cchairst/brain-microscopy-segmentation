"""Read images, masks and manifests; hash files."""

import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

from brainseg import IGNORE, LABELS


def digest(path):
    """SHA-256 of a file, read in 1 MB blocks."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_array(path, key='', channel=''):
    path = Path(path)
    if path.suffix.lower() == '.npy':
        a = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == '.mat':
        if not key:
            raise ValueError('MAT input requires an explicit mat_key in the manifest.')
        try:
            d = loadmat(path)
        except NotImplementedError as e:
            raise ValueError('MAT v7.3 is not supported; export a numeric 2D array to NPY or MAT v7.2.') from e
        if key not in d:
            raise ValueError(f'MAT key {key!r} missing from {path}.')
        a = np.asarray(d[key])
    else:
        a = np.asarray(Image.open(path))
    if channel != '':
        if a.ndim != 3:
            raise ValueError('Channel selection expects a channel-last H x W x C array.')
        a = a[..., int(channel)]
    if a.ndim != 2 or min(a.shape) < 2:
        raise ValueError(f'{path}: expected a 2D array; got {a.shape}. Select a numeric channel explicitly for RGB.')
    if not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all():
        raise ValueError(f'{path}: data must be finite and numeric.')
    return a


def read_mask(path, shape):
    a = read_array(path)
    if a.shape != shape or not np.isin(a, list(LABELS) + [IGNORE]).all():
        raise ValueError('Mask must match the image shape and contain only IDs 0,1,2,3,255.')
    return a.astype(np.uint8)


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        if not {'sample_id', 'section_id', 'split', 'image', 'mask'} <= set(reader.fieldnames or []):
            raise ValueError('Manifest requires sample_id,section_id,split,image,mask columns.')
        rows = list(reader)
    if not rows:
        raise ValueError('Manifest is empty.')
    groups, seen, images = {}, set(), set()
    for r in rows:
        if not r['sample_id'] or not r['section_id'] or r['split'] not in {'train', 'val', 'test'}:
            raise ValueError('Each row needs sample/section IDs and a train, val, or test split.')
        previous = groups.setdefault(r['sample_id'], r['split'])
        if previous != r['split']:
            raise ValueError(f'Leakage: sample {r["sample_id"]} occurs in multiple splits.')
        identity = (r['sample_id'], r['section_id'])
        if identity in seen:
            raise ValueError('Duplicate sample/section row.')
        seen.add(identity)
        for name in ['image', 'mask']:
            if r[name]:
                r[name] = str((path.parent / r[name]).resolve())
        if not r['image'] or r['image'] in images:
            raise ValueError('Every row must reference a distinct image path.')
        images.add(r['image'])
    return rows


def load_row(r):
    return read_array(r['image'], r.get('mat_key', ''), r.get('channel', ''))
