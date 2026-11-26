"""Training sets for a PS-OCT-only model: slide-deck sections labelled from optical and stable MRI estimates."""

import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes

from brainseg import IGNORE
from brainseg.methods.classical import segment
from brainseg.mri_labels import consensus
from brainseg.preprocess import unimaged

MODES = ('assisted', 'consensus')
OPTICAL, MRI = 1, 2  # source map: 0 background (both agree), 1 optical, 2 MRI fill, IGNORE not learned from


def stable(labels, steps):
    """True where a label equals its neighbours `steps[k]` pixels away in both directions along each axis k.

    With steps of one MRI voxel this checks that an MRI-derived label survives a one-voxel registration shift, as
    the six-shift audit of the MRI-assisted sections does on the MRI grid. Neighbours outside the array count as
    background.
    """
    out = np.ones(labels.shape, bool)
    for axis, step in enumerate(steps):
        for direction in (-1, 1):
            shifted = np.zeros_like(labels)
            if step < labels.shape[axis]:
                src, dst = [slice(None)] * labels.ndim, [slice(None)] * labels.ndim
                src[axis], dst[axis] = (slice(step, None), slice(None, -step))[::-direction]
                shifted[tuple(dst)] = labels[tuple(src)]
            out &= shifted == labels
    return out


def training_labels(otsu, mri, stable_mri, low_signal, inside, mode='assisted'):
    """Labels to learn from on one section, and the source of each (see OPTICAL, MRI).

    `consensus`: keep pixels where the MRI labels and otsu agree. `assisted`: keep otsu gray and white matter, then
    fill from MRI only where the optical image cannot decide: dark pixels `inside` the optical tissue outline (gaps such
    as ventricles; otsu's own fluid is mostly enclosed gel, so it is not trusted as such) and dropout (`low_signal`).
    A fill needs an MRI tissue label that is stable under a one-voxel shift, except MRI fluid in an enclosed optical
    gap: ventricles are thin next to an MRI voxel and rarely pass the shift check, and the two modalities agreeing
    stands in for it. Gel around the brain is never filled: the MRI outline passes it where registration is off.
    Background only where both say background. Everything else is IGNORE.
    """
    if mode not in MODES:
        raise ValueError(f'unknown mode: {mode}')
    labels = np.full(otsu.shape, IGNORE, np.uint8)
    source = labels.copy()
    if mode == 'consensus':
        labels = consensus(mri, otsu)
        source[labels == 0] = 0
        source[np.isin(labels, (1, 2, 3))] = OPTICAL
        return labels, source
    optical = np.isin(otsu, (1, 2)) & ~low_signal
    fluid = (mri == 3) & np.isin(otsu, (0, 3)) & ~low_signal
    fill = ~optical & np.isin(mri, (1, 2, 3)) & (inside | low_signal) & (stable_mri | fluid)
    background = (otsu == 0) & (mri == 0) & ~low_signal
    for mask, values, origin in ((background, 0, 0), (optical, otsu, OPTICAL), (fill, mri, MRI)):
        labels[mask] = values[mask] if isinstance(values, np.ndarray) else values
        source[mask] = origin
    return labels, source


def section_names(mapping, n):
    """Slide-deck index -> section name ('Slice_042') from the lab's slidedeck_slice_mapping.json, or the index."""
    if mapping is None:
        return [f'{i:03d}' for i in range(n)]
    names = {int(k): re.search(r'Slice_\d+', v).group(0) for k, v in json.loads(Path(mapping).read_text()).items()}
    return [names.get(i, f'{i:03d}') for i in range(n)]


def build(
    retardance,
    reflectivity,
    mri_labels,
    output,
    sample_id,
    axis,
    split='train',
    mode='assisted',
    floor=None,
    mapping=None,
    mri_voxel_mm=0.4,
):
    """Write a manifest-ready training set for one brain from its registered slide deck; return the report.

    `mri_labels` are the MRI labels warped onto the deck grid (notebook 02's `<subject>_labels_psoct.nii.gz`);
    `axis` is the deck axis that indexes sections; `floor` is the reflectivity value marking signal dropout.
    Sections without MRI tissue are skipped. Unimaged pixels are never labelled.
    """
    import nibabel as nib

    images = [nib.load(p) for p in (retardance, reflectivity, mri_labels)]
    if any(img.shape != images[0].shape or not np.allclose(img.affine, images[0].affine) for img in images):
        raise ValueError('Retardance, reflectivity and MRI labels must share the slide-deck grid.')
    zooms = nib.affines.voxel_sizes(images[0].affine)
    in_plane = np.delete(zooms, axis)
    if not np.isclose(*in_plane):
        raise ValueError(f'Expected square in-plane pixels, got {in_plane} mm.')
    steps = [max(1, round(mri_voxel_mm / z)) for z in zooms]
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    for sub in ('images', 'labels', 'sources'):
        (output / sub).mkdir(parents=True)

    mri = np.asarray(images[2].dataobj).astype(np.uint8)
    stable_mri = stable(mri, steps)
    ret, ref = (np.asarray(img.dataobj, dtype=np.float32) for img in images[:2])
    rows, sections = [], []
    for i, name in enumerate(section_names(mapping, mri.shape[axis])):
        m = np.take(mri, i, axis)
        if not np.isin(m, (1, 2, 3)).any():
            continue
        a = np.stack([np.take(ret, i, axis), np.take(ref, i, axis)], -1)
        otsu = segment(a, 'otsu', retardance=0, reflectivity=1, pixel_size=float(in_plane[0]))
        low = a[..., 1] == floor if floor is not None else np.zeros(m.shape, bool)
        inside = binary_fill_holes(np.isin(otsu, (1, 2, 3)))
        labels, source = training_labels(otsu, m, np.take(stable_mri, i, axis), low, inside, mode)
        outside = unimaged(a)
        labels[outside], source[outside] = IGNORE, IGNORE
        stem = f'{sample_id}_{name}'
        for c, contrast in enumerate(('ret', 'ref')):
            np.save(output / 'images' / f'{stem}_{contrast}.npy', a[..., c])
        Image.fromarray(labels).save(output / 'labels' / f'{stem}.png')
        Image.fromarray(source).save(output / 'sources' / f'{stem}.png')
        rows.append(
            {
                'sample_id': sample_id,
                'section_id': name,
                'split': split,
                'image': f'images/{stem}_ret.npy;images/{stem}_ref.npy',
                'mask': f'labels/{stem}.png',
                'pixel_size_mm': f'{in_plane[0]:.6g}',
            }
        )
        counts = {k: int((source == v).sum()) for k, v in (('background', 0), ('optical', OPTICAL), ('mri', MRI))}
        sections.append({'section_id': name, 'deck_index': i, **counts, 'ignored': int((source == IGNORE).sum())})
    with (output / 'manifest.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'section_id', 'split', 'image', 'mask', 'pixel_size_mm'])
        writer.writeheader()
        writer.writerows(rows)
    totals = {k: sum(s[k] for s in sections) for k in ('background', 'optical', 'mri', 'ignored')}
    report = {
        'sample_id': sample_id,
        'mode': mode,
        'split': split,
        'axis': axis,
        'floor': floor,
        'mri_voxel_mm': mri_voxel_mm,
        'stability_steps_px': steps,
        'sections': len(sections),
        'totals': totals,
        'per_section': sections,
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
