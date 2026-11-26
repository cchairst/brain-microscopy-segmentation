"""Whole-brain tissue estimates on the CMC ex vivo MRI grid."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
from scipy.ndimage import generate_binary_structure, maximum_filter, minimum_filter

from brainseg import IGNORE
from brainseg.methods.classical import segment
from brainseg.volume_viewer import write_viewer


def classify(ret, ref, mri, axis, spacing, floor):
    """Choose optical gray/white where supported, then use locally stable MRI fills."""
    inside = mri != 0
    tissue = np.where(inside, IGNORE, 0).astype(np.uint8)
    source = tissue.copy()
    optical_sections = np.moveaxis(tissue, axis, 0)
    source_sections = np.moveaxis(source, axis, 0)
    pixel_size = np.sqrt(np.prod(np.delete(spacing, axis)))
    for i, (r, f, brain) in enumerate(zip(*(np.moveaxis(a, axis, 0) for a in (ret, ref, inside)))):
        if not brain.any():
            continue
        base = segment(np.stack([r, f], axis=-1), 'otsu', reflectivity=1, pixel_size=pixel_size)
        keep = brain & np.isin(base, [1, 2]) & (r > 0) & (f > floor)
        optical_sections[i][keep] = base[keep]
        source_sections[i][keep] = 1

    # Equality with all six face neighbours, including background beyond the MRI volume.
    footprint = generate_binary_structure(3, 1)
    stable = minimum_filter(mri, footprint=footprint, mode='constant', cval=0) == maximum_filter(
        mri, footprint=footprint, mode='constant', cval=0
    )
    fill = (source == IGNORE) & np.isin(mri, [1, 2, 3]) & stable
    tissue[fill] = mri[fill]
    source[fill] = 2
    return tissue, source


# Every abstention has both an optical reason and an MRI reason.
REASON_NAMES = ('signal_mri_unknown', 'signal_mri_boundary', 'otsu_mri_unknown', 'otsu_mri_boundary')


def uncertainty_reasons(ret, ref, mri, source, floor):
    """Trace unresolved voxels through the optical gate and MRI fallback used in classify."""
    reason = np.zeros(source.shape, np.uint8)
    unresolved = source == IGNORE
    signal_passed = (ret[unresolved] > 0) & (ref[unresolved] > floor)
    mri_has_class = mri[unresolved] != IGNORE
    reason[unresolved] = 1 + 2 * signal_passed + mri_has_class
    return reason


def slice_diagnostics(reason, source, axis):
    """Count paired failure reasons and pick a sizable slice with the highest unresolved fraction."""
    sources = np.moveaxis(source, axis, 0)
    brain = np.count_nonzero(sources, axis=(1, 2))
    rows = []
    for i, plane in enumerate(np.moveaxis(reason, axis, 0)):
        counts = np.bincount(plane.ravel(), minlength=5)[1:]
        unresolved = int(counts.sum())
        rows.append([i, int(brain[i]), unresolved, unresolved / brain[i] if brain[i] else 0.0, *counts.tolist()])
    eligible = [row for row in rows if row[1] >= brain.max() / 2]
    priority = max(eligible, key=lambda row: row[3])[0]
    return rows, priority


def run(subject, reflectivity, retardance, mri_labels, output):
    """Process all slices, including blank end slices, and save on the MRI affine."""
    ref, ret, labels = (nib.load(p) for p in (reflectivity, retardance, mri_labels))
    if subject not in ('Moe', 'Vlad'):
        raise ValueError('Subject must be Moe or Vlad.')
    if (
        len(labels.shape) != 3
        or len(ref.shape) != 3
        or ref.shape != ret.shape
        or not np.allclose(ref.affine, ret.affine)
    ):
        raise ValueError('Expected a 3D MRI reference and two optical contrasts on the same 3D grid.')
    axis, floor = (1, 0) if subject == 'Moe' else (0, 32.5)
    if ref.shape == labels.shape and np.allclose(ref.affine, labels.affine):
        offset = (0, 0, 0)
    elif (
        subject == 'Moe'
        and ref.shape == (346, 346, 112)
        and labels.shape == (160, 200, 112)
        and np.allclose(ref.affine[:3, :3], labels.affine[:3, :3], atol=1e-5)
    ):
        offset = (90, 70, 0)
    else:
        raise ValueError('Expected the section contrasts in MRI space or the documented padded Moe grid.')
    crop = tuple(slice(start, start + n) for start, n in zip(offset, labels.shape))
    contrasts = [img.get_fdata(dtype=np.float32)[crop] for img in (ret, ref)]
    if any(not np.isfinite(a).all() for a in contrasts):
        raise ValueError('Registered MRI-space optical contrasts must contain finite values.')
    mri = np.asanyarray(labels.dataobj)
    if not np.isin(mri, [0, 1, 2, 3, IGNORE]).all() or not (mri != 0).any():
        raise ValueError('Expected nonempty semantic MRI labels from notebook 02.')
    tissue, source = classify(*contrasts, mri.astype(np.uint8), axis, labels.header.get_zooms()[:3], floor)
    reason = uncertainty_reasons(*contrasts, mri, source, floor)
    rows, priority = slice_diagnostics(reason, source, axis)
    report = {
        'priority_slice_index': priority,
        'uncertain_by_reason': dict(zip(REASON_NAMES, np.bincount(reason.ravel(), minlength=5)[1:].tolist())),
        'subject': subject,
        'shape': list(labels.shape),
        'section_axis': axis,
        'crop_offset': offset,
        'slices_processed': labels.shape[axis],
        'optical_voxels': int((source == 1).sum()),
        'mri_fill_voxels': int((source == 2).sum()),
        'uncertain_voxels': int((source == IGNORE).sum()),
        'brain_voxels': int(np.count_nonzero(mri)),
    }
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / 'volume'
        stage.mkdir()
        for name, arr in [('tissue', tissue), ('source', source), ('uncertainty_reason', reason)]:
            header = labels.header.copy()
            header.set_data_dtype(np.uint8)
            nib.save(nib.Nifti1Image(arr, labels.affine, header), stage / f'{name}.nii.gz')
        with (stage / 'slice_diagnostics.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['slice_index', 'brain_voxels', 'uncertain_voxels', 'uncertain_fraction', *REASON_NAMES])
            writer.writerows(rows)
        write_viewer(stage / 'viewer.html', subject, axis, contrasts[1], tissue, source, reason, priority)
        (stage / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        stage.rename(output)
    return report
