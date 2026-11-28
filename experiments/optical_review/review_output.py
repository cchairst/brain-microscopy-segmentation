"""Build paired optical/segmentation viewers and a fixed-scale review plate."""

import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Patch, Rectangle
from PIL import Image
from scipy.signal import convolve2d

PALETTE = {1: '#f1a64d', 2: '#5ca7f1', 3: '#b785d8', 255: '#898e96'}
FLAG = '#00ffd2'
SUBJECTS = [('Moe', 1, 'coronal'), ('Vlad', 0, 'sagittal')]


def rgb(hexcode):
    return np.frombuffer(bytes.fromhex(hexcode[1:]), dtype=np.uint8)


def display_plane(volume, axis, index):
    return np.flipud(np.take(volume, index, axis=axis).T)


def bundle(inputs, current, output):
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.7), gridspec_kw={'width_ratios': [1.52, 1, 1]})
    fig.subplots_adjust(left=0.025, right=0.985, top=0.84, bottom=0.17, hspace=0.28, wspace=0.10)
    fig.suptitle('Where the optical signal warrants a second look', x=0.035, y=0.985, ha='left', size=20, weight='bold')
    fig.text(
        0.035,
        0.921,
        'Whole slice at left. Both close-ups show the boxed 32 × 32 voxels with the same local contrast.',
        size=11,
    )
    template = Path(__file__).with_name('viewer_template.html').read_text()
    for row, (subject, axis, orientation) in enumerate(SUBJECTS):
        current_subject = current / subject
        tissue_im = nib.load(current_subject / 'tissue.nii.gz')
        tissue = np.asarray(tissue_im.dataobj)
        src = np.asarray(nib.load(current_subject / 'source.nii.gz').dataobj)
        flags_im = nib.load(output / f'{subject}_review_flags.nii.gz')
        flags = np.asarray(flags_im.dataobj)
        assert np.allclose(flags_im.affine, tissue_im.affine)
        assert flags.shape == tissue.shape and not np.any((flags != 0) & (src != 255))
        ref = nib.load(inputs / f'{subject}Ref.nii.gz').get_fdata(dtype=np.float32)
        if ref.shape != tissue.shape and subject == 'Moe':
            ref = ref[90:250, 70:270, :]
        assert ref.shape == tissue.shape
        low, high = np.percentile(ref[ref > 0], [1, 99])
        gray = np.round(255 * np.clip((ref - low) / (high - low), 0, 1)).astype(np.uint8)
        counts = np.moveaxis(flags, axis, 0).sum((1, 2)).tolist()
        best = int(np.argmax(counts))
        frames = []
        for idx in range(len(counts)):
            pixels = np.stack([display_plane(v, axis, idx) for v in (gray, tissue, flags)], axis=-1)
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format='PNG')
            frames.append(base64.b64encode(buffer.getvalue()).decode('ascii'))
        html = (
            template.replace('@SUBJECT@', subject)
            .replace('@PLANE@', orientation)
            .replace('@START@', str(best))
            .replace('@COUNTS@', json.dumps(counts))
            .replace('@FRAMES@', json.dumps(frames))
        )
        (output / f'{subject}_review.html').write_text(html)
        raw, labels, flagged = [display_plane(v, axis, best) for v in (gray, tissue, flags)]
        width = 32
        density = convolve2d(flagged.astype(np.int32), np.ones((width, width), np.int32), mode='valid')
        y, x = np.unravel_index(density.argmax(), density.shape)
        window = (slice(y, y + width), slice(x, x + width))
        overlay = np.repeat(raw[:, :, None], 3, axis=2)
        for code, color in PALETTE.items():
            mask = labels == code
            overlay[mask] = np.round(0.45 * overlay[mask] + 0.55 * rgb(color))
        native_crop = display_plane(ref, axis, best)[window]
        lo, hi = np.percentile(native_crop, [2, 98])
        detail = np.round(255 * np.clip((native_crop - lo) / max(hi - lo, 1e-6), 0, 1)).astype(np.uint8)
        closeup = np.repeat(detail[:, :, None], 3, axis=2)
        closeup[flagged[window] > 0] = rgb(FLAG)
        ax, rawax, flagax = axes[row]
        ax.imshow(overlay, interpolation='nearest')
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), width, width, linewidth=1.5, edgecolor=FLAG, facecolor='none'))
        rawax.imshow(detail, vmin=0, vmax=255, cmap='gray', interpolation='nearest')
        flagax.imshow(closeup, interpolation='nearest')
        ax.set_title(f'{subject}   ·   {orientation} {best}', fontsize=12)
        rawax.set_title('Optical reflectivity', fontsize=12)
        flagax.set_title(f'{int(flagged[window].sum())} flagged voxels in crop', fontsize=12)
        for panel in axes[row]:
            panel.set_axis_off()
    legend = [
        (PALETTE[1], 'Gray matter'),
        (PALETTE[2], 'White matter'),
        (PALETTE[3], 'Possible fluid'),
        (PALETTE[255], 'Uncertain'),
        (FLAG, 'Review flag'),
    ]
    fig.legend(
        handles=[Patch(color=color, label=label) for color, label in legend],
        loc='lower center',
        bbox_to_anchor=(0.5, 0.06),
        ncol=5,
        frameon=False,
    )
    fig.text(
        0.5,
        0.027,
        'Each animal’s slice contains its highest flag count. '
        'Cyan pixels had been left unresolved in the segmentation.',
        ha='center',
        size=10,
        color='#414754',
    )
    fig.savefig(output / 'BCM_optical_review.png', dpi=175)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('inputs', 'current', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    args = p.parse_args()
    bundle(args.inputs, args.current, args.output)
