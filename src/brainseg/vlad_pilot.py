"""Compare single-contrast and multicontrast 3D Atropos on aligned Vlad PS-OCT.

This is an exploratory segmentation pilot without anatomical ground truth.
"""

import importlib.metadata
import json
import tempfile
from pathlib import Path

import ants
import nibabel as nib
import numpy as np
from ants.internal import get_lib_fn, process_arguments
from scipy.ndimage import binary_closing, binary_fill_holes, label
from sklearn.metrics import adjusted_rand_score

from brainseg.io import digest
from brainseg.preprocess import percentile_bounds, scale


def otsu(values):
    h, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w = np.cumsum(h).astype(float)
    s = np.cumsum(h * centers)
    denominator = w * (w[-1] - w)
    score = np.divide((s[-1] * w - s * w[-1]) ** 2, denominator, out=np.zeros_like(w), where=denominator > 0)
    return float(centers[np.argmax(score)])


def make_image(a, reference):
    return ants.from_numpy(
        np.asarray(a), origin=reference.origin, spacing=reference.spacing, direction=reference.direction
    )


def fit_atropos(inputs, mask, iterations, likelihood='Gaussian'):
    # ANTsPy 0.6.3 drops kwargs in its multivariate branch. Pass all arguments
    # directly to the bundled Atropos entry point so seed/model/verbosity apply.
    with tempfile.TemporaryDirectory(prefix='vlad_atropos_') as temporary:
        folder = Path(temporary)
        for i, image in enumerate(inputs):
            ants.image_write(image, str(folder / f'input{i}.nii.gz'))
        ants.image_write(mask, str(folder / 'mask.nii.gz'))
        args = {
            'd': 3,
            'x': str(folder / 'mask.nii.gz'),
            'i': 'KMeans[3]',
            'm': '[0.2,1x1x1]',
            'c': f'[{iterations},0]',
            'r': 0,
            'k': likelihood,
            'v': 1,
            'o': f'[{folder}/classes.nii.gz,{folder}/prob%02d.nii.gz]',
        }
        for i in range(len(inputs)):
            args[f'a-MULTINAME-{i}'] = str(folder / f'input{i}.nii.gz')
        status = get_lib_fn('Atropos')(process_arguments(args))
        if status:
            raise RuntimeError(f'Atropos failed with status {status}; inspect diagnostic log.')
        return {
            'segmentation': ants.image_read(str(folder / 'classes.nii.gz')),
            'probabilityimages': [ants.image_read(str(folder / f'prob{i:02d}.nii.gz')) for i in [1, 2, 3]],
        }


def run(data, output, factor=2, iterations=10, mask_path=None):
    if factor < 1 or iterations < 1:
        raise ValueError('factor and iterations must be positive integers')
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = [data / f'{c}_slide_deck_in_MRI.nii.gz' for c in ['Ref', 'Cro', 'Ret']]
    headers = [nib.load(p) for p in paths]
    for image in headers:
        if len(image.shape) != 3 or image.shape != headers[0].shape or not np.allclose(image.affine, headers[0].affine):
            raise ValueError('All inputs must be 3D and have identical grids and affines.')
        if not np.isfinite(np.asanyarray(image.dataobj)).all():
            raise ValueError('Non-finite input values')
    original = [ants.image_read(str(p)).clone('float') for p in paths]
    spacing = tuple(float(s * factor) for s in original[0].spacing)
    images = [
        ants.resample_image(im, spacing, use_voxels=False, interp_type=0) if factor > 1 else im for im in original
    ]
    reference = images[0]
    arrays = [im.numpy() for im in images]
    threshold = None
    mask_source = None
    if mask_path:
        header = nib.load(mask_path)
        if header.shape != headers[0].shape or not np.allclose(header.affine, headers[0].affine):
            raise ValueError('Supplied mask grid and affine must match the native volumes.')
        raw_mask = np.asanyarray(header.dataobj)
        if not np.isfinite(raw_mask).all() or not np.isin(raw_mask, [0, 1]).all():
            raise ValueError('Supplied mask must be finite and binary.')
        mask_image = ants.image_read(str(mask_path))
        mask_image = ants.resample_image_to_target(mask_image, reference, interp_type='nearestNeighbor')
        supplied_mask = mask_image.numpy() > 0
        # The MRI mask may include brain locations without optical coverage.
        mask = supplied_mask & (arrays[0] > 0)
        mask_source = {
            'name': Path(mask_path).name,
            'sha256': digest(mask_path),
            'mask_voxels': int(supplied_mask.sum()),
            'optical_coverage_fraction': float(mask.sum() / max(supplied_mask.sum(), 1)),
        }
        mask_status = (
            'Lab-provided MRI brain mask intersected with positive reflectivity support; not expert tissue ground truth'
        )
    else:
        threshold = otsu(arrays[0])
        candidate = binary_closing(arrays[0] > threshold, iterations=1)
        components, _ = label(candidate)
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        if sizes.max() < 100:
            raise ValueError('Provisional mask too small')
        mask = binary_fill_holes(components == sizes.argmax())
        mask_status = 'UNREVIEWED Otsu threshold, 3D closing, largest component, hole filling'
    if mask.sum() < 100:
        raise ValueError('Segmentation mask too small')
    mask_im = make_image(mask.astype(np.uint8), reference)
    ants.image_write(mask_im, str(output / 'provisional_foreground_mask.nii.gz'))
    normalized = []
    normalization = []
    for a in arrays:
        lo, hi = percentile_bounds(a, mask)
        if hi <= lo:
            raise ValueError('Insufficient intensity contrast in provisional mask')
        # Atropos Gaussian likelihood rejects covariance determinants < 1e-6.
        # Equal 0..100 scaling avoids that absolute-unit floor for 3D features;
        # it does not add noise, change relative channel weighting, or alter data geometry.
        normalized.append(make_image((100 * scale(a, lo, hi) + 0.1).astype(np.float32), reference))
        normalization.append([float(lo), float(hi)])
    labels = {}
    runs = {}
    for name, inputs in [('reflectivity_only', normalized[:1]), ('three_contrasts', normalized)]:
        print(f'Running {name}: {reference.shape}, {int(mask.sum())} masked voxels', flush=True)
        result = fit_atropos(inputs, mask_im, iterations)
        seg = result['segmentation'].numpy().astype(np.uint8)
        probabilities = np.stack([p.numpy() for p in result['probabilityimages']], axis=-1)
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(-1)[mask], 1, atol=1e-4):
            raise ValueError('Posterior probabilities failed validation')
        if not np.isin(seg, [0, 1, 2, 3]).all() or np.any(seg[~mask] != 0) or np.any(seg[mask] == 0):
            raise ValueError('Labels violate the segmentation mask')
        labels[name] = seg
        ants.image_write(result['segmentation'], str(output / f'{name}_classes.nii.gz'))
        for i, probability in enumerate(result['probabilityimages'], 1):
            ants.image_write(probability, str(output / f'{name}_class{i}_posterior.nii.gz'))
        counts = [int((seg == i).sum()) for i in [1, 2, 3]]
        means = [[float(a[seg == i].mean()) if counts[i - 1] else None for a in arrays] for i in [1, 2, 3]]
        runs[name] = {
            'class_voxels': counts,
            'class_fraction_of_mask': [c / int(mask.sum()) for c in counts],
            'class_mean_raw_Ref_Cro_Ret': means,
            'mean_max_posterior': float(probabilities.max(-1)[mask].mean()),
            'fraction_max_posterior_below_0_6': float((probabilities.max(-1)[mask] < 0.6).mean()),
        }
    # ARI is label-permutation invariant. It measures agreement, NOT tissue accuracy.
    agreement = float(adjusted_rand_score(labels['reflectivity_only'][mask], labels['three_contrasts'][mask]))
    report = {
        'sample': 'Vlad',
        'stage': 'exploratory MRI-aligned PS-OCT 3D pilot',
        'source_project': 'https://brainlife.io/project/6a8335945c0941a11f67cc4e/files',
        'source_folder': 'Derivatives/Vlad/PS-OCT/Enface/Registration',
        'input_files': [{'name': p.name, 'sha256': digest(p)} for p in paths],
        'native_shape': list(headers[0].shape),
        'native_spacing_mm': [float(v) for v in headers[0].header.get_zooms()],
        'processed_shape': list(reference.shape),
        'processed_spacing_mm': list(reference.spacing),
        'factor': factor,
        'foreground_threshold_raw_reflectivity': threshold,
        'foreground_voxels': int(mask.sum()),
        'foreground_fraction_of_grid': float(mask.mean()),
        'mask_status': mask_status,
        'mask_source': mask_source,
        'normalization_p1_p99_Ref_Cro_Ret': normalization,
        'normalized_intensity_range': [0.1, 100.1],
        'algorithm': {
            'tool': 'ANTs Atropos',
            'classes': 3,
            'iterations': iterations,
            'mrf': 0.2,
            'radius': '1x1x1',
            'random_seed_flag': 0,
        },
        'versions': {
            p: importlib.metadata.version(p) for p in ['antspyx', 'numpy', 'scipy', 'nibabel', 'scikit-learn']
        },
        'runs': runs,
        'inter_method_adjusted_rand_index': agreement,
        'reference_tissue_masks': None,
        'tissue_accuracy_dice_iou': None,
        'limitations': [
            'Intensity classes have no validated gray/white/fluid labels.',
            'Foreground mask and positive-reflectivity support need visual review; tears can remain included.',
            'MRI-aligned optical derivatives are resampled; compare with native optical images before conclusions.',
            'One sample, one pilot setting, no cross-sample performance estimate.',
            'Posterior confidence and method agreement are not accuracy.',
        ],
    }
    report['quality_flags'] = [
        f'{name}: dominant class covers {max(v["class_fraction_of_mask"]):.1%} of support; '
        'tissue separation needs review'
        for name, v in runs.items()
        if max(v['class_fraction_of_mask']) > 0.8
    ]
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    plot(arrays[0], mask, labels, output / 'vlad_3d_pilot.png')
    print(json.dumps({'runs': runs, 'inter_method_adjusted_rand_index': agreement}, indent=2), flush=True)
    return report


def plot(reflectivity, mask, labels, path):
    """Inputs, support mask and both Atropos runs along the three voxel axes, at the slices with most support."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    palette = ListedColormap(['black', '#478ee6', '#e69b32', '#46c373'])
    indices = []
    for axis in range(3):
        totals = mask.sum(axis=tuple(i for i in range(3) if i != axis))
        indices.append(int(np.argmax(totals)))
    fig, axes = plt.subplots(3, 4, figsize=(13, 10), facecolor='white', constrained_layout=True)
    titles = ['Reflectivity', 'Segmentation support mask', 'Atropos: reflectivity', 'Atropos: 3 contrasts']
    for axis, index in enumerate(indices):
        for col, a in enumerate([reflectivity, mask, labels['reflectivity_only'], labels['three_contrasts']]):
            section = np.rot90(np.take(a, index, axis=axis))
            ax = axes[axis, col]
            if col == 0:
                vmin, vmax = np.percentile(reflectivity[mask], 1), np.percentile(reflectivity[mask], 99)
                ax.imshow(section, cmap='gray', vmin=vmin, vmax=vmax)
            elif col == 1:
                ax.imshow(section, cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(section, cmap=palette, vmin=0, vmax=3, interpolation='nearest')
            ax.set_title(titles[col] if axis == 0 else f'Voxel axis {axis} | index {index}', fontsize=10)
            ax.axis('off')
    fig.suptitle(
        'Vlad PS-OCT | 3D MRI-tool pilot\n'
        'Blue / orange / green: increasing reflectivity groups, not verified tissue labels',
        fontsize=15,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
