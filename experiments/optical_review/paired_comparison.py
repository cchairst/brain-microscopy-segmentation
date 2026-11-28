"""Compare both classifiers on exactly the same held-out voxels."""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from threshold_trial import propose


def run(inputs, current, output):
    rows = []
    for subject, axis in [('Moe', 1), ('Vlad', 0)]:
        target = nib.load(inputs / f'{subject}_MRI_reference.nii.gz')
        reference = np.asarray(target.dataobj)
        source = np.asarray(nib.load(current / subject / 'source.nii.gz').dataobj)
        signals = [nib.load(inputs / f'{subject}{name}.nii.gz').get_fdata(dtype=np.float32) for name in ('Ret', 'Ref')]
        if subject == 'Moe' and signals[0].shape != source.shape:
            signals = [signal[90:250, 70:270, :] for signal in signals]
        _, relaxed = propose(*signals, axis, target.header.get_zooms(), 32.5 if subject == 'Vlad' else 0)
        forest = np.asarray(nib.load(output / f'{subject}_experimental_candidates.nii.gz').dataobj)
        selected = forest != 0
        assert np.all(source[selected] == 255) and np.all(relaxed[selected] > 0)
        mask = selected & np.isin(reference, (1, 2, 3))
        forest_ok = forest[mask] == reference[mask]
        relaxed_ok = relaxed[mask] == reference[mask]
        rows.append(
            dict(
                subject=subject,
                scorable=int(mask.sum()),
                forest_matches=int(forest_ok.sum()),
                threshold_matches=int(relaxed_ok.sum()),
                always_gray_matches=int((reference[mask] == 1).sum()),
                both_match=int((forest_ok & relaxed_ok).sum()),
                forest_only=int((forest_ok & ~relaxed_ok).sum()),
                threshold_only=int((~forest_ok & relaxed_ok).sum()),
                neither_matches=int((~forest_ok & ~relaxed_ok).sum()),
            )
        )
    with (output / 'paired_results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('inputs', 'current', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    run(args.inputs, args.current, args.output)
