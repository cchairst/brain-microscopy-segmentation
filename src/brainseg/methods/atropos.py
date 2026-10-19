"""Apply the ANTs Atropos medical-image segmentation tool to PS-OCT.

Classes are intensity clusters, not validated anatomical tissue labels.
"""

import importlib.metadata
import json
from pathlib import Path

import ants
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing, binary_fill_holes, label

from brainseg.io import digest
from brainseg.preprocess import percentile_bounds, scale


def segment(image_path, output, mask_path=None, max_size=1100, classes=3, threshold=20, mrf=0.2):
    """Segment a 2D grayscale PNG/JPEG or a 2D/3D NIfTI into `classes` intensity groups; write outputs and a report."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    nifti = image_path.lower().endswith(('.nii', '.nii.gz'))
    original_shape = None
    if nifti:
        image = ants.image_read(image_path).clone('float')
        array = image.numpy()
    else:
        Image.MAX_IMAGE_PIXELS = 200000000
        im = Image.open(image_path)
        if im.mode != 'L':
            raise ValueError('Use an explicitly selected scalar PS-OCT contrast, not a colorized RGB image.')
        original_shape = list(im.size)
        im.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        array = np.asarray(im).astype(np.float32)
        image = ants.from_numpy(array)
    if image.dimension not in (2, 3) or not np.isfinite(array).all():
        raise ValueError('Expected a finite 2D section or 3D volume.')
    if mask_path:
        if mask_path.lower().endswith(('.nii', '.nii.gz')):
            mask_image = ants.image_read(mask_path)
            if not ants.image_physical_space_consistency(image, mask_image):
                raise ValueError('Mask and image must have identical grid and physical geometry.')
            mask = mask_image.numpy() > 0
        else:
            mask = np.asarray(Image.open(mask_path)) > 0
        if mask.shape != array.shape:
            raise ValueError('Mask must match processed image shape.')
    else:
        if nifti:
            raise ValueError('NIfTI inputs require a reviewed mask; no arbitrary intensity threshold is inferred.')
        candidate = binary_fill_holes(binary_closing(array > threshold, iterations=2))
        components, _ = label(candidate)
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        if not sizes.max():
            raise ValueError('Automatic mask is empty.')
        mask = components == sizes.argmax()
    if np.count_nonzero(mask) < 100:
        raise ValueError('Mask too small.')
    lo, hi = percentile_bounds(array, mask)
    if hi <= lo:
        raise ValueError('No usable intensity contrast inside mask.')
    normalized = scale(array, lo, hi).astype(np.float32)
    # Preserve physical metadata for numeric NIfTI inputs.
    image = ants.from_numpy(normalized, origin=image.origin, spacing=image.spacing, direction=image.direction)
    mask_image = ants.from_numpy(
        mask.astype(np.uint8), origin=image.origin, spacing=image.spacing, direction=image.direction
    )
    radius = 'x'.join(['1'] * image.dimension)
    result = ants.atropos(a=image, x=mask_image, i=f'Kmeans[{classes}]', m=f'[{mrf},{radius}]', c='[10,0]', r=1)
    segmentation = result['segmentation'].numpy().astype(np.uint8)
    ants.image_write(result['segmentation'], str(out / 'intensity_classes.nii.gz'))
    ants.image_write(mask_image, str(out / 'foreground_mask.nii.gz'))
    for i, posterior in enumerate(result['probabilityimages'], 1):
        ants.image_write(posterior, str(out / f'class_{i}_posterior.nii.gz'))
    counts = {str(i): int(np.count_nonzero(segmentation == i)) for i in range(1, classes + 1)}
    report = {
        'tool': 'ANTs Atropos',
        'antspyx_version': importlib.metadata.version('antspyx'),
        'source_sha256': digest(image_path),
        'source_dimensions_wh': original_shape,
        'processed_shape': list(array.shape),
        'mask': 'provided' if mask_path else 'UNREVIEWED automatic threshold, closing, hole fill, largest component',
        'classes': classes,
        'mrf': mrf,
        'iterations': 10,
        'class_counts': counts,
        'class_mean_intensity': {
            str(i): float(array[segmentation == i].mean()) if counts[str(i)] else None for i in range(1, classes + 1)
        },
        'interpretation': 'Classes ordered by input intensity; anatomical mapping is NOT validated.',
        'accuracy': None,
        'reference_masks_available': False,
    }
    if image.dimension == 2:
        if nifti:
            gray = np.repeat((normalized * 255).astype(np.uint8)[..., None], 3, -1)
        else:
            gray = np.repeat(np.clip(array, 0, 255).astype(np.uint8)[..., None], 3, -1)
        colors = np.asarray([[0, 0, 0], [70, 130, 230], [230, 155, 50], [70, 195, 115]], dtype=np.uint8)
        if classes > 3:
            extra = np.random.default_rng(42).integers(60, 230, (classes - 3, 3), dtype=np.uint8)
            colors = np.vstack([colors, extra])
        colored = colors[segmentation]
        overlay = gray.copy()
        overlay[mask] = (0.55 * gray[mask] + 0.45 * colored[mask]).astype(np.uint8)
        Image.fromarray(segmentation).save(out / 'intensity_classes.png')
        h, w = array.shape
        panel = Image.new('RGB', (w * 3, h + 70), 'white')
        draw = ImageDraw.Draw(panel)
        panels = [
            (gray, 'PS-OCT reflectivity'),
            (colored, 'Atropos intensity classes'),
            (overlay, 'Overlay - unvalidated'),
        ]
        for i, (a, title) in enumerate(panels):
            panel.paste(Image.fromarray(a), (i * w, 35))
            draw.text((i * w + 10, 10), title, fill='black')
        legend = 'Blue: low intensity | Orange: middle | Green: high | No tissue labels or accuracy established'
        draw.text((10, h + 43), legend, fill='black')
        panel.save(out / 'preview.png')
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
