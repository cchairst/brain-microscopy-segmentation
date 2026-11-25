import numpy as np
import pytest

from brainseg import IGNORE
from brainseg.cli import main

nib = pytest.importorskip('nibabel')


def test_fill_tracks_origin_and_preserves_unresolved_tissue():
    from brainseg.assisted import fill

    optical = np.array([0, 0, 0, 1, 2, 3, IGNORE, 0], dtype=np.uint8)
    mri = np.array([0, 1, IGNORE, 2, 1, 1, 3, 3], dtype=np.uint8)
    floor = np.array([False] * 4 + [True] + [False] * 3)
    tissue, source = fill(optical, mri, floor)
    assert tissue.tolist() == [0, 1, IGNORE, 1, 1, 3, 3, 3]
    assert source.tolist() == [0, 2, IGNORE, 1, 2, 1, 2, 2]


@pytest.mark.parametrize('x_sign', [-1, 1])
def test_inverse_warp_in_fsl_mm_with_padded_labels(tmp_path, x_sign):
    from brainseg.assisted import project

    affine = np.diag([2.0 * x_sign, 1.0, 1.0, 1.0])
    shape = (10, 8, 9)

    def image(a, matrix=affine):
        return nib.Nifti1Image(a, matrix)

    field = np.zeros((*shape, 3), np.float32)
    field[..., 0] = 2  # one x voxel in FSL mm; sign depends on the voxel handedness
    warp = image(field)
    warp.header.set_intent(2006)
    deck = image(np.zeros(shape, np.float32))
    labels = np.zeros((6, 7, 9), np.uint8)
    labels[2, 3, 4] = 3
    section_affine = affine.copy()
    section_affine[1, 3] = 4
    section = image(np.zeros((10, 1, 9), np.float32), section_affine)
    mapped, _ = project(section, deck, warp, image(labels), deck, (1, 1, 0))
    np.testing.assert_array_equal(np.argwhere(mapped), [[3 - x_sign, 0, 4]])
    with pytest.raises(ValueError, match='offset'):
        project(section, deck, warp, image(labels), deck, None)


def test_assisted_cli_preserves_section_geometry(tmp_path):
    affine = np.diag([1.0, 0.055, 0.055, 1.0])

    def save(name, data, matrix=affine):
        path = tmp_path / (name + '.nii.gz')
        image = nib.Nifti1Image(data, matrix)
        if name == 'warp':
            image.header.set_intent(2006)
        nib.save(image, path)
        return str(path)

    section_affine = affine.copy()
    section_affine[0, 3] = 1
    section = save('section', np.full((1, 6, 7), 32.5, np.float32), section_affine)
    shape = (3, 6, 7)
    registered = save('registered', np.full(shape, 32.5, np.float32))
    warp = save('warp', np.zeros((*shape, 3), np.float32))
    labels = save('labels', np.full(shape, 2, np.uint8))
    out = tmp_path / 'out'
    args = [
        'assist-section',
        '--reflectivity',
        section,
        '--retardance',
        section,
        '--floor',
        '32.5',
        '--deck',
        registered,
        '--registered-ref',
        registered,
        '--warp',
        warp,
        '--mri-labels',
        labels,
        '--output',
        str(out),
    ]
    main(args)
    for filename in ['tissue', 'source']:
        result = nib.load(out / (filename + '.nii.gz'))
        np.testing.assert_array_equal(result.get_fdata(), np.full((1, 6, 7), 2))
        np.testing.assert_allclose(result.affine, section_affine)
        assert result.get_data_dtype() == np.uint8
    with pytest.raises(FileExistsError):
        main(args)
