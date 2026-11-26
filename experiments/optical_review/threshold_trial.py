"""Run from a checkout with PYTHONPATH=src; see README for the input layout."""

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

from brainseg.methods.classical import segment
from brainseg.preprocess import normalize, otsu


def propose(ret, ref, axis, spacing, floor):
    """Optical-only proposals. MRI classes and mask do not enter either gate."""
    baseline = np.zeros(ref.shape, np.uint8)
    loose = np.zeros_like(baseline)
    pixel_size = np.sqrt(np.prod(np.delete(spacing, axis)))
    planes = [np.moveaxis(a, axis, 0) for a in (ret, ref, baseline, loose)]
    for r, f, seeds, new in zip(*planes):
        imaged = f > 0
        raw = segment(np.stack([r, f], -1), 'otsu', reflectivity=1, pixel_size=pixel_size)
        good = (r > 0) & (f > floor)
        accepted = np.isin(raw, [1, 2]) & good
        seeds[accepted] = raw[accepted]
        if not np.any(seeds == 1) or not np.any(seeds == 2):
            continue
        norm = normalize(f)
        cut = otsu(norm[imaged])
        low = norm[imaged] <= cut
        if low.mean() < 0.2:
            cut = otsu(norm[imaged & (norm > cut)])
        # Recover only pixels originally called background in the band below Otsu.
        band = good & (raw == 0) & (norm <= cut) & (norm > cut - 0.05)
        scaled_ret = normalize(r, accepted)
        new[band] = 1 + (scaled_ret[band] > otsu(scaled_ret[accepted]))
    return baseline, loose


def bracket_votes(seeds, proposed):
    """Count axes where the nearest optical labels on both sides support the proposal."""
    padded = np.pad(seeds, 2)
    votes = np.zeros_like(seeds)
    for dim in range(3):
        sides = []
        for sign in (-1, 1):
            nearest = np.zeros_like(seeds)
            for dist in (1, 2):
                shifts = [2, 2, 2]
                shifts[dim] += sign * dist
                view = padded[tuple(slice(start, start + n) for start, n in zip(shifts, seeds.shape))]
                np.copyto(nearest, view, where=nearest == 0)
            sides.append(nearest)
        votes += (proposed > 0) & (sides[0] == proposed) & (sides[1] == proposed)
    return votes


def score(pred, mri, source, region, subject, method, subset):
    added = (pred != 0) & region
    inside = mri != 0
    review = added & (source == 255)
    scorable = review & np.isin(mri, [1, 2, 3])
    row = dict(
        subject=subject,
        method=method,
        subset=subset,
        proposed=int(added.sum()),
        outside_MRI=int((added & ~inside).sum()),
        outside_beyond_one_voxel=int((added & ~binary_dilation(inside)).sum()),
        inside_MRI=int((added & inside).sum()),
        previously_uncertain=int(review.sum()),
        uncertain_without_MRI_class=int((review & (mri == 255)).sum()),
    )
    for k, name in [(1, 'gray'), (2, 'white')]:
        support = scorable & (pred == k)
        row[name + '_scorable'] = int(support.sum())
        row[name + '_matches'] = int((support & (mri == k)).sum())
    return row


def run(inputs, current, out):
    rows = []
    for s, axis in [('Moe', 1), ('Vlad', 0)]:
        idx = json.loads((current / s / 'report.json').read_text())['priority_slice_index']
        target = nib.load(inputs / f'{s}_MRI_reference.nii.gz')
        registered = [nib.load(inputs / f'{s}{c}.nii.gz') for c in ('Ret', 'Ref')]
        assert registered[0].shape == registered[1].shape and np.allclose(registered[0].affine, registered[1].affine)
        mri = np.asarray(target.dataobj)
        crop = (slice(90, 250), slice(70, 270), slice(None)) if s == 'Moe' else (slice(None),) * 3
        ret, ref = [im.get_fdata(dtype=np.float32)[crop] for im in registered]
        assert ret.shape == mri.shape
        source = np.asarray(nib.load(current / s / 'source.nii.gz').dataobj)
        tissue = np.asarray(nib.load(current / s / 'tissue.nii.gz').dataobj)
        baseline, loose = propose(ret, ref, axis, target.header.get_zooms(), 32.5 if s == 'Vlad' else 0)
        assert np.array_equal(np.where(mri != 0, baseline, 0), np.where(source == 1, tissue, 0))
        votes = bracket_votes(baseline, loose)
        # Keep the actionable review layer binary. A flag carries no tissue class.
        flags = ((source == 255) & (votes >= 2)).astype(np.uint8)
        header = target.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(flags, target.affine, header), out / f'{s}_review_flags.nii.gz')
        focus = np.zeros(mri.shape, bool)
        np.moveaxis(focus, axis, 0)[idx] = True
        for threshold in (0, 2, 3):
            proposed = np.where(votes >= threshold, loose, 0).astype(np.uint8)
            for region, subset in [(np.ones(mri.shape, bool), 'whole'), (focus, f'index_{idx}')]:
                rows.append(score(proposed, mri, source, region, s, f'{threshold}_axes', subset))

        # Only for the review figure; this is an experimental overlay, not a tissue mask.
        def plane(a):
            return np.flipud(np.take(a, idx, axis=axis).T)

        np.savez_compressed(
            out / f'{s}_review.npz',
            reflectivity=plane(ref),
            baseline=plane(baseline),
            relaxed=plane(loose),
            votes=plane(votes),
            mri=plane(mri),
            source=plane(source),
        )
    with (out / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('inputs', 'current', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    run(a.inputs, a.current, a.output)
