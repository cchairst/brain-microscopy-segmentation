"""MRI-derived pseudo-labels: segment the ex vivo MRI with Atropos and warp the labels into PS-OCT space.

These are references made from a different modality at 0.4 mm, not hand labels. Agreement with them measures
agreement with the MRI, not accuracy.
"""

import numpy as np
from scipy.ndimage import binary_erosion

from brainseg import IGNORE
from brainseg.preprocess import percentile_bounds, scale


def segment(fa, t2, md, mask, spacing, iterations=10):
    """Three-class Atropos on FA, T2 and log-MD inside `mask`; return (classes 1-3, posteriors of shape (3, ...)).

    `spacing` is the voxel size in mm; Atropos's MRF smoothing depends on it.
    """
    import ants

    from brainseg.vlad_pilot import fit_atropos, make_image

    reference = ants.from_numpy(np.asarray(fa, dtype=np.float32), spacing=tuple(float(s) for s in spacing))

    def channel(a):
        lo, hi = percentile_bounds(a, mask)
        # Atropos has an absolute covariance floor, so keep values well away from 0..1 (see vlad_pilot.py).
        return make_image((scale(a, lo, hi) * 100 + 0.1).astype(np.float32), reference)

    log_md = np.log(np.clip(md, 1e-5, None))
    result = fit_atropos(
        [channel(fa), channel(t2), channel(log_md)], make_image(mask.astype(np.uint8), reference), iterations
    )
    classes = result['segmentation'].numpy().astype(np.uint8)
    posteriors = np.stack([p.numpy() for p in result['probabilityimages']])
    return classes, posteriors


def name_classes(classes, fa, md, fluid_md_ratio=1.5):
    """Map Atropos classes 1-3 to label IDs.

    The highest-MD class is fluid only if its median MD is at least `fluid_md_ratio` times the median MD of all
    classified voxels; some brains (e.g. Moe) have no fluid class. Among the rest, the lowest-FA class is gray matter
    and the others are white matter.
    """
    stats = {k: (fa[classes == k].mean(), np.median(md[classes == k])) for k in (1, 2, 3)}
    candidate = max(stats, key=lambda k: stats[k][1])
    fluid = candidate if stats[candidate][1] >= fluid_md_ratio * np.median(md[classes > 0]) else None
    gray = min((k for k in stats if k != fluid), key=lambda k: stats[k][0])
    labels = np.zeros_like(classes, dtype=np.uint8)
    for k in (1, 2, 3):
        labels[classes == k] = 3 if k == fluid else 1 if k == gray else 2
    return labels


def mark_uncertain(labels, fa, mask, posteriors, shell=3, fluid_max_fa=0.2, min_posterior=0.6):
    """Clean up partial volume: fluid must have low FA.

    High-FA fluid voxels deep in the brain (white matter next to ventricles, e.g. corpus callosum) become white
    matter; those within `shell` voxels of the brain surface become IGNORE. Voxels whose largest posterior is below
    `min_posterior` also become IGNORE.
    """
    labels = labels.copy()
    surface = mask & ~binary_erosion(mask, iterations=shell)
    high_fa_fluid = (labels == 3) & (fa >= fluid_max_fa)
    labels[high_fa_fluid & ~surface] = 2
    labels[high_fa_fluid & surface] = IGNORE
    labels[mask & (posteriors.max(0) < min_posterior)] = IGNORE
    return labels


def label_mri(fa, t2, md, mask, spacing, **kwargs):
    """Segment, name and clean up in one call; keyword arguments go to `mark_uncertain`."""
    classes, posteriors = segment(fa, t2, md, mask, spacing)
    return mark_uncertain(name_classes(classes, fa, md), fa, mask, posteriors, **kwargs)


def warp(labels, source, warp_path, reference_path):
    """Resample `labels` into PS-OCT space through an FSL FNIRT warp, with nearest-neighbour interpolation.

    `source` is the nibabel image the warp was computed from (the MRI FA); `labels` must be on its grid.
    `reference_path` is any image on the PS-OCT target grid, e.g. a `*_slide_deck.nii.gz`.
    """
    import nibabel as nib
    from fsl.data.image import Image
    from fsl.transform.fnirt import readFnirt
    from fsl.transform.nonlinear import applyDeformation

    if labels.shape != source.shape[:3]:
        raise ValueError(f'labels {labels.shape} are not on the warp source grid {source.shape[:3]}')
    src = Image(nib.Nifti1Image(labels.astype(np.uint8), None, source.header))
    ref = Image(reference_path)
    field = readFnirt(warp_path, src, ref)
    return applyDeformation(src, field, ref=ref, order=0, mode='constant', cval=0).astype(np.uint8)
