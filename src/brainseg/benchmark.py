"""Evaluate a trained model on a manifest split, and a synthetic end-to-end demo."""

import csv
import inspect
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

from brainseg import IGNORE, LABELS
from brainseg.io import digest, image_digest, load_row, read_manifest, read_mask
from brainseg.methods.baselines import predict_array, train
from brainseg.methods.classical import segment
from brainseg.metrics import hd95, metrics


def run_predictions(manifest, model_path, split, output):
    """Predict a manifest split with a trained model and score it."""
    artifact = joblib.load(model_path)  # Only load trusted artifacts; joblib can execute code.
    return evaluate(
        manifest,
        split,
        output,
        artifact['method'],
        lambda a: predict_array(artifact, a),
        model_sha256=digest(model_path),
        train_sample_ids=artifact['train_sample_ids'],
    )


def run_classical(manifest, split, output, method='gmm', **settings):
    """Segment a manifest split with an unsupervised classical method and score it; record every setting."""
    defaults = {k: p.default for k, p in inspect.signature(segment).parameters.items() if k != 'a'}
    settings = {**defaults, **settings, 'method': method}
    return evaluate(manifest, split, output, method, lambda a: segment(a, **settings), settings=settings)


def evaluate(manifest, split, output, method, predict, model_sha256=None, train_sample_ids=(), settings=None):
    """Run `predict` on each H x W x C section of a split, save label PNGs, and score annotated sections."""
    rows = read_manifest(manifest)
    selected = [r for r in rows if r['split'] == split]
    if not selected:
        raise ValueError(f'No {split} rows.')
    if split != 'train' and {r['sample_id'] for r in selected} & set(train_sample_ids):
        raise ValueError('Evaluation samples overlap model training samples.')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    n = len(LABELS)
    cms, hds, sections, pooled = {}, {}, [], np.zeros((n, n), dtype=np.int64)
    for i, r in enumerate(selected):
        a = load_row(r)
        prediction = predict(a)
        filename = f'{i:05d}_labels.png'
        Image.fromarray(prediction).save(out / filename)
        record = {
            'sample_id': r['sample_id'],
            'section_id': r['section_id'],
            'prediction': filename,
            'image_sha256': image_digest(r['image']),
            'annotated': bool(r['mask']),
        }
        if r['mask']:
            y = read_mask(r['mask'], a.shape[:2])
            valid = (y != IGNORE) & ~np.isnan(a).any(-1)  # unimaged pixels are not trained on, so not scored
            cm = confusion_matrix(y[valid], prediction[valid], labels=list(LABELS))
            pooled += cm
            cms.setdefault(r['sample_id'], np.zeros((n, n), dtype=np.int64))[:] += cm
            record['metrics'] = metrics(cm)
            m = record['metrics']
            m['hd95_px'] = {LABELS[c]: hd95(prediction, y, c, valid) for c in LABELS if c}
            # Optional pixel_size_mm column: HD95 in um (null without it).
            um = 1000 * float(r['pixel_size_mm']) if r.get('pixel_size_mm') else None
            m['hd95_um'] = {LABELS[c]: hd95(prediction, y, c, valid, um) if um else None for c in LABELS if c}
            hds.setdefault(r['sample_id'], []).append({unit: m[unit] for unit in ('hd95_px', 'hd95_um')})
        sections.append(record)
    samples = {s: metrics(cm) for s, cm in cms.items()}
    for s, per_section in hds.items():  # HD95 does not pool across sections: average the section values
        for unit in ('hd95_px', 'hd95_um'):
            by_class = {k: [h[unit][k] for h in per_section if h[unit][k] is not None] for k in per_section[0][unit]}
            samples[s][unit] = {k: float(np.mean(v)) if v else None for k, v in by_class.items()}
    values = [v['macro_tissue']['dice'] for v in samples.values() if v['macro_tissue']['dice'] is not None]
    report = {
        'method': method,
        'split': split,
        'model_sha256': model_sha256,
        'manifest_sha256': digest(manifest),
        'label_ids': LABELS,
        'sections': sections,
        'samples': samples,
        'pooled': metrics(pooled) if cms else None,
        'mean_sample_macro_dice': float(np.mean(values)) if values else None,
        'note': 'Undefined metrics are null. Tissue macro and volume fractions cover gray matter, white matter '
        'and fluid; damage is scored per class only. Unannotated pixels (255) are ignored. '
        'HD95 is in pixels, and in um for rows with pixel_size_mm; per-sample HD95 is the mean of its sections.',
    }
    if settings is not None:
        report['settings'] = settings
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def demo(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[:96, :96]
    rows = []
    for i, split in enumerate(['train', 'train', 'train', 'val', 'test', 'test']):
        radius = np.hypot(xx - 48 - rng.uniform(-3, 3), yy - 48)
        mask = np.zeros((96, 96), dtype=np.uint8)
        mask[radius < 40] = 1
        mask[radius < 28] = 2
        mask[radius < 10] = 3
        image = np.asarray([0.05, 0.4, 0.8, 0.15])[mask] + rng.normal(0, 0.025, mask.shape)
        np.save(root / f'brain{i}.npy', image.astype(np.float32))
        Image.fromarray(mask).save(root / f'brain{i}_mask.png')
        rows.append(
            {
                'sample_id': f'synthetic_brain{i}',
                'section_id': '001',
                'split': split,
                'image': f'brain{i}.npy',
                'mask': f'brain{i}_mask.png',
            }
        )
    manifest = root / 'manifest.csv'
    with manifest.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for method in ['kmeans', 'rf']:
        path = root / f'{method}.joblib'
        train(manifest, method, path, pixels=4000)
        report = run_predictions(manifest, path, 'test', root / method)
        print(f'SYNTHETIC DEMO ONLY {method}: mean sample Dice = {report["mean_sample_macro_dice"]:.4f}')
