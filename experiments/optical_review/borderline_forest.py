"""Cross-animal evaluation of an optical classifier trained on near-threshold voxels."""

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestClassifier
from threshold_trial import propose, score


def optical_features(ret, ref, baseline, loose, axis):
    seed = baseline > 0
    band = loose > 0
    normalized = []
    for signal in (ret, ref):
        scaled = np.zeros(signal.shape, np.float32)
        for a, b, z in zip(*[np.moveaxis(v, axis, 0) for v in (signal, seed, scaled)]):
            if not b.any():
                continue
            lo, hi = np.percentile(a[b], [10, 90])
            z[:] = np.clip((a - lo) / max(hi - lo, 1.0e-6), -3, 3)
        normalized.append(scaled)
    features = [a[band] for a in normalized]
    for sigma in (1, 3, 6):
        for a in [*normalized, seed, baseline == 2]:
            features.append(gaussian_filter(a.astype(np.float32), sigma)[band])
    return np.column_stack(features)


def run(inputs, current, out):
    cases = {}
    for s, axis in [('Moe', 1), ('Vlad', 0)]:
        im = nib.load(inputs / f'{s}_MRI_reference.nii.gz')
        mri = np.asarray(im.dataobj)
        crop = (slice(90, 250), slice(70, 270), slice(None)) if s == 'Moe' else (slice(None),) * 3
        ret, ref = [nib.load(inputs / f'{s}{c}.nii.gz').get_fdata(dtype=np.float32)[crop] for c in ('Ret', 'Ref')]
        seeds, loose = propose(ret, ref, axis, im.header.get_zooms(), 32.5 if s == 'Vlad' else 0)
        band = loose > 0
        x = optical_features(ret, ref, seeds, loose, axis)
        source = np.asarray(nib.load(current / s / 'source.nii.gz').dataobj)
        cases[s] = (x, mri, source, band)
    rows = []
    for train, test, axis in [('Moe', 'Vlad', 0), ('Vlad', 'Moe', 1)]:
        idx = json.loads((current / test / 'report.json').read_text())['priority_slice_index']
        x, mri, _, band = cases[train]
        y = mri[band]
        pool = np.flatnonzero(y != 255)
        ids = np.random.default_rng(11).choice(pool, min(160000, len(pool)), replace=False)
        model = RandomForestClassifier(n_estimators=100, max_depth=18, min_samples_leaf=10, n_jobs=4, random_state=11)
        model.fit(x[ids], y[ids])
        xt, ytest, src, band = cases[test]
        prob = model.predict_proba(xt)
        winner = model.classes_[prob.argmax(axis=1)]
        high = (prob.max(axis=1) >= 0.95) & np.isin(winner, [1, 2])
        pred = np.zeros(src.shape, np.uint8)
        pred[band] = np.where(high, winner, 0)
        focus = np.zeros(src.shape, bool)
        np.moveaxis(focus, axis, 0)[idx] = True
        for region, subset in [(np.ones(src.shape, bool), 'whole'), (focus, f'index_{idx}')]:
            rows.append(score(pred, ytest, src, region, test, 'other_subject_RF_p95', subset))
        nib.save(
            nib.Nifti1Image(
                np.where(src == 255, pred, 0).astype(np.uint8), nib.load(current / test / 'source.nii.gz').affine
            ),
            out / f'{test}_experimental_candidates.nii.gz',
        )
    with (out / 'forest_results.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ['inputs', 'current', 'output']:
        p.add_argument('--' + k, required=True, type=Path)
    args = p.parse_args()
    run(args.inputs, args.current, args.output)
