"""Apply the ANTs Atropos medical-image segmentation tool to PS-OCT.

Classes are intensity clusters, not validated anatomical tissue labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import importlib.metadata

import ants
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing, binary_fill_holes, label


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', required=True, help='2D grayscale PNG/JPEG or 2D/3D NIfTI')
    p.add_argument('--mask', help='Reviewed binary mask on the processed grid; NIfTI or lossless image')
    p.add_argument('--output', required=True)
    p.add_argument('--max-size', type=int, default=1100, help='Max JPEG/PNG side for a downsampled pilot')
    p.add_argument('--classes', type=int, default=3)
    p.add_argument('--threshold', type=float, default=20, help='Unreviewed foreground mask threshold on 0-255 image inputs')
    p.add_argument('--mrf', type=float, default=.2)
    args = p.parse_args()
    if args.classes < 2 or args.max_size < 32 or args.mrf < 0:
        p.error('classes >= 2, max-size >= 32, and mrf >= 0 are required')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    nifti = args.image.lower().endswith(('.nii', '.nii.gz'))
    original_shape = None
    if nifti:
        image = ants.image_read(args.image).clone('float')
        array = image.numpy()
    else:
        Image.MAX_IMAGE_PIXELS = 200000000
        im = Image.open(args.image)
        if im.mode != 'L':
            raise ValueError('Use an explicitly selected scalar PS-OCT contrast, not a colorized RGB image.')
        original_shape = list(im.size)
        im.thumbnail((args.max_size, args.max_size), Image.Resampling.LANCZOS)
        array = np.asarray(im).astype(np.float32)
        image = ants.from_numpy(array)
    if image.dimension not in (2, 3) or not np.isfinite(array).all():
        raise ValueError('Expected a finite 2D section or 3D volume.')
    if args.mask:
        if args.mask.lower().endswith(('.nii', '.nii.gz')):
            mask_image = ants.image_read(args.mask)
            if not ants.image_physical_space_consistency(image, mask_image):
                raise ValueError('Mask and image must have identical grid and physical geometry.')
            mask = mask_image.numpy() > 0
        else:
            mask = np.asarray(Image.open(args.mask)) > 0
        if mask.shape != array.shape:
            raise ValueError('Mask must match processed image shape.')
    else:
        if nifti:
            raise ValueError('NIfTI inputs require a reviewed mask; no arbitrary intensity threshold is inferred.')
        candidate = binary_fill_holes(binary_closing(array > args.threshold, iterations=2))
        components, _ = label(candidate)
        sizes = np.bincount(components.ravel()); sizes[0] = 0
        if not sizes.max():
            raise ValueError('Automatic mask is empty.')
        mask = components == sizes.argmax()
    if np.count_nonzero(mask) < 100:
        raise ValueError('Mask too small.')
    lo, hi = np.percentile(array[mask], [1, 99])
    if hi <= lo:
        raise ValueError('No usable intensity contrast inside mask.')
    normalized = np.clip((array - lo) / (hi - lo), 0, 1).astype(np.float32)
    # Preserve physical metadata for numeric NIfTI inputs.
    image = ants.from_numpy(normalized, origin=image.origin, spacing=image.spacing, direction=image.direction)
    mask_image = ants.from_numpy(mask.astype(np.uint8), origin=image.origin, spacing=image.spacing, direction=image.direction)
    radius = 'x'.join(['1'] * image.dimension)
    result = ants.atropos(a=image, x=mask_image, i=f'Kmeans[{args.classes}]',
                          m=f'[{args.mrf},{radius}]', c='[10,0]', r=1)
    segmentation = result['segmentation'].numpy().astype(np.uint8)
    ants.image_write(result['segmentation'], str(out / 'intensity_classes.nii.gz'))
    ants.image_write(mask_image, str(out / 'foreground_mask.nii.gz'))
    for i, posterior in enumerate(result['probabilityimages'], 1):
        ants.image_write(posterior, str(out / f'class_{i}_posterior.nii.gz'))
    counts = {str(i): int(np.count_nonzero(segmentation == i)) for i in range(1, args.classes + 1)}
    report = {'tool': 'ANTs Atropos', 'antspyx_version': importlib.metadata.version('antspyx'),
              'source_sha256': hashlib.sha256(Path(args.image).read_bytes()).hexdigest(),
              'source_dimensions_wh': original_shape, 'processed_shape': list(array.shape),
              'mask': 'provided' if args.mask else 'UNREVIEWED automatic threshold, closing, hole fill, largest component',
              'classes': args.classes, 'mrf': args.mrf, 'iterations': 10,
              'class_counts': counts, 'class_mean_intensity': {
                  str(i): float(array[segmentation == i].mean()) if counts[str(i)] else None
                  for i in range(1, args.classes + 1)},
              'interpretation': 'Classes ordered by input intensity; anatomical mapping is NOT validated.',
              'accuracy': None, 'reference_masks_available': False}
    if image.dimension == 2:
        gray = np.repeat(np.clip(array, 0, 255).astype(np.uint8)[..., None], 3, -1) if not nifti else np.repeat((normalized * 255).astype(np.uint8)[..., None], 3, -1)
        colors = np.asarray([[0, 0, 0], [70, 130, 230], [230, 155, 50], [70, 195, 115]], dtype=np.uint8)
        if args.classes > 3:
            colors = np.vstack([colors, np.random.default_rng(42).integers(60, 230, (args.classes - 3, 3), dtype=np.uint8)])
        colored = colors[segmentation]
        overlay = gray.copy(); overlay[mask] = (.55 * gray[mask] + .45 * colored[mask]).astype(np.uint8)
        Image.fromarray(segmentation).save(out / 'intensity_classes.png')
        h, w = array.shape
        panel = Image.new('RGB', (w * 3, h + 70), 'white')
        draw = ImageDraw.Draw(panel)
        for i, (a, title) in enumerate([(gray, 'PS-OCT reflectivity'), (colored, 'Atropos intensity classes'), (overlay, 'Overlay - unvalidated')]):
            panel.paste(Image.fromarray(a), (i * w, 35)); draw.text((i * w + 10, 10), title, fill='black')
        draw.text((10, h + 43), 'Blue: low intensity | Orange: middle | Green: high | No tissue labels or accuracy established', fill='black')
        panel.save(out / 'preview.png')
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
