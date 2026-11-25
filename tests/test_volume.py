import json

import numpy as np
import pytest

from brainseg.cli import main

nib = pytest.importorskip('nibabel')


def test_volume_fills_only_stable_mri_and_keeps_all_slices(tmp_path):
    shape = (9, 11, 13)
    mri = np.zeros(shape, np.uint8)
    mri[1:-1, 1:-1, 1:-1] = 1
    mri[4:, 1:-1, 1:-1] = 2
    mri[6, 8, 8] = 255
    affine = np.diag([0.4, 0.4, 0.4, 1])
    for name, values in [('mri', mri), ('ret', np.zeros(shape, np.float32)), ('ref', np.zeros(shape, np.float32))]:
        nib.save(nib.Nifti1Image(values, affine), tmp_path / f'{name}.nii.gz')
    out = tmp_path / 'results'
    args = [
        'assist-volume',
        '--subject',
        'Vlad',
        '--reflectivity',
        str(tmp_path / 'ref.nii.gz'),
        '--retardance',
        str(tmp_path / 'ret.nii.gz'),
        '--mri-labels',
        str(tmp_path / 'mri.nii.gz'),
        '--output',
        str(out),
    ]
    main(args)
    tissue = nib.load(out / 'tissue.nii.gz')
    source = np.asarray(nib.load(out / 'source.nii.gz').dataobj)
    assert tissue.shape == mri.shape and np.allclose(tissue.affine, affine)
    labels = np.asarray(tissue.dataobj)
    assert labels[2, 5, 5] == 1 and labels[6, 5, 5] == 2
    assert labels[3, 5, 5] == labels[4, 5, 5] == labels[6, 8, 8] == 255
    np.testing.assert_array_equal(labels == 0, mri == 0)
    np.testing.assert_array_equal(source == 2, np.isin(labels, [1, 2]))
    assert json.loads((out / 'report.json').read_text())['slices_processed'] == shape[0]
    html = (out / 'viewer.html').read_text()
    assert 'Vlad' in html and 'Sagittal' in html and 'FRAMES' not in html
    with pytest.raises(FileExistsError):
        main(args)
    shifted = affine.copy()
    shifted[1, 3] = 10
    nib.save(nib.Nifti1Image(mri, shifted), tmp_path / 'ret.nii.gz')
    with pytest.raises(ValueError, match='grid'):
        main(args)
