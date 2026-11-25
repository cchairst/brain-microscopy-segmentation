"""MRI-assisted labeling on a native CMC PS-OCT section."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from brainseg import IGNORE
from brainseg.methods.classical import segment


def fsl_affine(image):
    """Map voxels to FSL scaled millimetres, including FSL's neurological x flip."""
    spacing = image.header.get_zooms()[:3]
    affine = np.diag([*spacing, 1.0])
    if np.linalg.det(image.affine[:3, :3]) > 0:
        affine[0, 0] *= -1
        affine[0, 3] = (image.shape[0] - 1) * spacing[0]
    return affine


def project(section, deck, warp, labels, registered_ref, offset):
    """Invert the lab's relative FNIRT displacement at each section pixel."""
    if len(deck.shape) != 3 or len(warp.shape) != 4 or warp.shape[-1] != 3:
        raise ValueError('Expected a 3D slide deck and a 3-component warp field.')
    if warp.header.get_intent()[0] != 'fnirt disp field':
        raise ValueError('Expected the PSOCT_to_MRI FNIRT displacement field.')
    if registered_ref.shape != warp.shape[:3] or not np.allclose(registered_ref.affine, warp.affine):
        raise ValueError('Registered reflectivity must share the warp grid.')
    if len(labels.shape) != 3 or not np.allclose(labels.affine[:3, :3], warp.affine[:3, :3], atol=1e-5):
        raise ValueError('MRI label axes and spacing must match the warp grid.')
    if offset is None:
        if labels.shape != warp.shape[:3] or not np.allclose(labels.affine, warp.affine):
            raise ValueError('Different MRI and warp grids require an explicit voxel offset.')
        offset = (0, 0, 0)
    offset = np.asarray(offset)
    if np.any(offset < 0) or np.any(offset + labels.shape > warp.shape[:3]):
        raise ValueError('The MRI label volume must fit inside the warp grid at the given offset.')
    values = np.asanyarray(labels.dataobj)
    if not np.isin(values, [0, 1, 2, 3, IGNORE]).all():
        raise ValueError('Expected semantic MRI tissue labels 0, 1, 2, 3 or 255.')
    field = warp.get_fdata(dtype=np.float32)
    if not np.isfinite(field).all():
        raise ValueError('Nonfinite displacement field.')
    to_deck = fsl_affine(deck) @ np.linalg.inv(deck.affine) @ section.affine
    pixels = np.indices(section.shape).reshape(3, -1)
    target = nib.affines.apply_affine(to_deck, pixels.T).T
    to_voxel = np.linalg.inv(fsl_affine(warp))
    center = tuple(np.array(warp.shape[:3]) // 2)
    position = target - field[center][:, None]
    for _ in range(40):
        vox = nib.affines.apply_affine(to_voxel, position.T).T
        shift = np.stack([map_coordinates(field[..., k], vox, order=1, mode='nearest') for k in range(3)])
        residual = target - position - shift
        if np.max(np.abs(residual)) < 1e-6:
            break
        position += 0.8 * residual
    # Discard unconverged samples. Out-of-volume sampling uses constant background.
    valid = np.max(np.abs(residual), axis=0) < 1e-3
    mapped = map_coordinates(values.astype(np.uint8), vox - offset[:, None], order=0)
    mapped[~valid] = 0
    check = map_coordinates(registered_ref.get_fdata(dtype=np.float32), vox, order=1)
    check[~valid] = np.nan
    return mapped.reshape(section.shape), check.reshape(section.shape)


def fill(optical, mri, low_signal):
    missing = (mri != 0) & ((optical == 0) | (optical == IGNORE) | low_signal)
    supplied = missing & np.isin(mri, [1, 2, 3])
    tissue = optical.copy()
    tissue[missing] = IGNORE
    tissue[supplied] = mri[supplied]
    source = np.zeros_like(tissue)
    source[np.isin(tissue, [1, 2, 3, 4])] = 1
    source[supplied] = 2
    source[tissue == IGNORE] = IGNORE
    return tissue, source


def run(reflectivity, retardance, deck, warp, mri_labels, registered_ref, output, offset=None, floor=None):
    section, ret = nib.load(reflectivity), nib.load(retardance)
    if len(section.shape) != 3 or section.shape.count(1) != 1:
        raise ValueError('Expected one native 2D section stored as a singleton-axis 3D NIfTI.')
    if section.shape != ret.shape or not np.allclose(section.affine, ret.affine):
        raise ValueError('The optical section contrasts must share a grid.')
    spacing = [step for n, step in zip(section.shape, section.header.get_zooms()) if n > 1]
    ref = section.get_fdata(dtype=np.float32).squeeze()
    stack = np.stack([ret.get_fdata(dtype=np.float32).squeeze(), ref], axis=-1)
    optical = segment(stack, 'otsu', reflectivity=1, pixel_size=np.sqrt(np.prod(spacing)))
    mapped, check = project(
        section, nib.load(deck), nib.load(warp), nib.load(mri_labels), nib.load(registered_ref), offset
    )
    mri = mapped.squeeze()
    tissue, source = fill(optical, mri, (ref == floor) if floor is not None else False)

    # Reproject the lab's registered reflectivity for an independent coordinate check.
    smooth = gaussian_filter(ref, 3)
    check = check.squeeze()
    common = (smooth > 25) & (check > 25)
    corr = None
    if common.sum() > 1 and smooth[common].std() > 0 and check[common].std() > 0:
        corr = float(np.corrcoef(smooth[common], check[common])[0, 1])
    report = {
        'reflectivity_correlation': corr,
        'floor': floor,
        'mri_voxel_offset': offset,
        'optical_pixels': int((source == 1).sum()),
        'mri_filled_pixels': int((source == 2).sum()),
        'uncertain_pixels': int((source == IGNORE).sum()),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    with TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary) / 'section'
        staging.mkdir()
        for name, array in [('tissue', tissue), ('source', source)]:
            hdr = section.header.copy()
            hdr.set_data_dtype(np.uint8)
            image = nib.Nifti1Image(array.reshape(section.shape), section.affine, hdr)
            nib.save(image, staging / f'{name}.nii.gz')
        (staging / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        staging.rename(output)
    return report
