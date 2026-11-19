import nibabel as nib
import numpy as np
import pytest

from brainseg import IGNORE
from brainseg.mri_labels import label_mri, mark_uncertain, name_classes, warp


def phantom():
    """Ball of gray matter around a white-matter shell around a fluid cavity; FA, T2 and MD follow the tissue."""
    z, y, x = np.mgrid[:24, :24, :24]
    r = np.sqrt((x - 12) ** 2 + (y - 12) ** 2 + (z - 12) ** 2)
    truth = np.zeros((24, 24, 24), np.uint8)
    truth[r < 10] = 1
    truth[r < 7] = 2
    truth[r < 3.5] = 3
    rng = np.random.default_rng(0)
    fa = np.choose(truth, [0, 0.25, 0.6, 0.02]) + rng.normal(0, 0.02, truth.shape)
    t2 = np.choose(truth, [0, 60, 45, 120]) + rng.normal(0, 2, truth.shape)
    md = np.choose(truth, [1e-5, 3e-4, 2.5e-4, 3e-3]) * rng.uniform(0.95, 1.05, truth.shape)
    return truth, fa, t2, md, truth > 0


def test_name_classes_by_md_then_fa():
    truth, fa, _, md, _ = phantom()
    shuffled = np.choose(truth, [0, 3, 1, 2]).astype(np.uint8)  # Atropos class IDs in arbitrary order
    np.testing.assert_array_equal(name_classes(shuffled, fa, md), truth)


def test_name_classes_without_fluid_gives_two_white_matter_classes():
    truth, fa, _, md, _ = phantom()
    md = np.where(truth == 3, 3e-4, md)  # no elevated diffusivity anywhere
    fa = np.where(truth == 3, 0.8, fa)  # third class is a second, more anisotropic white matter
    named = name_classes(truth, fa, md)
    np.testing.assert_array_equal(named, np.choose(truth, [0, 1, 2, 2]))


def test_mark_uncertain_moves_deep_high_fa_fluid_and_drops_surface():
    labels = np.zeros((9, 9, 9), np.uint8)
    labels[1:8, 1:8, 1:8] = 3
    fa = np.full(labels.shape, 0.5)
    fa[4, 4, 5] = 0.1  # one low-FA voxel stays fluid
    mask = labels > 0
    posteriors = np.ones((3,) + labels.shape)
    posteriors[:, 4, 4, 3] = 0.4  # one low-confidence voxel
    out = mark_uncertain(labels, fa, mask, posteriors, shell=2)
    assert out[1, 1, 1] == IGNORE  # surface shell
    assert out[4, 4, 4] == 2  # deep, high FA
    assert out[4, 4, 5] == 3  # deep, low FA
    assert out[4, 4, 3] == IGNORE  # low posterior
    assert out[0, 0, 0] == 0


def test_label_mri_recovers_phantom():
    pytest.importorskip('ants')
    truth, fa, t2, md, mask = phantom()
    labels = label_mri(fa, t2, md, mask, spacing=(0.4, 0.4, 0.4), shell=1)
    scored = labels != IGNORE
    assert scored.mean() > 0.8
    assert (labels[scored] == truth[scored]).mean() > 0.95


def test_warp_with_zero_displacement_is_identity(tmp_path):
    pytest.importorskip('fsl')
    affine = np.diag([0.4, 0.4, 0.4, 1])
    labels = np.random.default_rng(0).integers(0, 4, (8, 9, 10)).astype(np.uint8)
    source = nib.Nifti1Image(labels.astype(np.float32), affine)
    reference = tmp_path / 'reference.nii.gz'
    nib.save(source, reference)
    field = nib.Nifti1Image(np.zeros((8, 9, 10, 3), np.float32), affine)
    field.header.set_intent(2006)  # FSL_FNIRT_DISPLACEMENT_FIELD, relative displacements in mm
    nib.save(field, tmp_path / 'warp.nii.gz')
    np.testing.assert_array_equal(warp(labels, source, tmp_path / 'warp.nii.gz', reference), labels)
