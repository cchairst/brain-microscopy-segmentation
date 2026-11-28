"""Audit how one-voxel shifts of the MRI reference affect optical candidates."""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np


def audit(pred, reference):
    if pred.shape != reference.shape or pred.ndim != 3:
        raise ValueError('Expected two 3D arrays on the same grid.')
    if not np.isin(pred, [0, 1, 2]).all() or not np.isin(reference, [0, 1, 2, 3, 255]).all():
        raise ValueError('Expected gray/white candidate IDs and semantic MRI label IDs.')
    # Use identical centers for every shift, including at image edges.
    known = (pred != 0) & np.isin(reference, [1, 2, 3])
    same = known & (pred == reference)
    padded = np.pad(reference, 1, constant_values=0)
    nearby = np.zeros(pred.shape, bool)
    rows = []
    for dim in range(3):
        for step in [-1, 1]:
            start = [1, 1, 1]
            start[dim] += step
            offset = tuple(slice(i, i + n) for i, n in zip(start, pred.shape))
            neighbor = padded[offset]
            match = known & (pred == neighbor)
            nearby |= match
            rows.append(
                dict(
                    axis=dim,
                    step=step,
                    scorable=int(known.sum()),
                    original_matches=int(same.sum()),
                    shifted_matches=int(match.sum()),
                    gains=int((match & ~same).sum()),
                    losses=int((same & ~match).sum()),
                )
            )
    codes = np.zeros(pred.shape, np.uint8)
    codes[(pred > 0) & ~known] = 4
    codes[same] = 1
    codes[known & ~same & nearby] = 2
    codes[known & ~same & ~nearby] = 3
    return codes, rows


def run(candidates, mri, output):
    pred, ref = (nib.load(path) for path in (candidates, mri))
    if pred.shape != ref.shape or not np.allclose(pred.affine, ref.affine, rtol=0, atol=1e-4):
        raise ValueError('Candidate and MRI grids differ. Check registration before scoring.')
    codes, rows = audit(np.asanyarray(pred.dataobj), np.asanyarray(ref.dataobj))
    output.mkdir(parents=True, exist_ok=True)
    header = ref.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(codes, ref.affine, header), output / 'boundary_audit.nii.gz')
    with (output / 'shift_scores.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return np.bincount(codes.ravel(), minlength=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidates', 'mri', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    counts = run(args.candidates, args.mri, args.output)
    print('Exact, adjacent only, absent locally, unscorable:', *counts[1:])
