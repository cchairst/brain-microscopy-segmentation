"""Evaluate a trained model on a manifest split, and a synthetic end-to-end demo."""

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

from brainseg import IGNORE, LABELS
from brainseg.io import digest, load_row, read_manifest, read_mask
from brainseg.methods.baselines import predict_array, train
from brainseg.metrics import metrics


def run_predictions(manifest, model_path, split, output):
    rows = read_manifest(manifest)
    artifact = joblib.load(model_path)  # Only load trusted artifacts; joblib can execute code.
    selected = [r for r in rows if r['split'] == split]
    if not selected:
        raise ValueError(f'No {split} rows.')
    if split != 'train' and {r['sample_id'] for r in selected} & set(artifact['train_sample_ids']):
        raise ValueError('Evaluation samples overlap model training samples.')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    cms, sections, pooled = {}, [], np.zeros((4, 4), dtype=np.int64)
    for i, r in enumerate(selected):
        a = load_row(r)
        prediction = predict_array(artifact, a)
        filename = f'{i:05d}_labels.png'
        Image.fromarray(prediction).save(out / filename)
        record = {
            'sample_id': r['sample_id'],
            'section_id': r['section_id'],
            'prediction': filename,
            'image_sha256': digest(r['image']),
            'annotated': bool(r['mask']),
        }
        if r['mask']:
            y = read_mask(r['mask'], a.shape)
            valid = y != IGNORE
            cm = confusion_matrix(y[valid], prediction[valid], labels=list(LABELS))
            pooled += cm
            cms.setdefault(r['sample_id'], np.zeros((4, 4), dtype=np.int64))[:] += cm
            record['metrics'] = metrics(cm)
        sections.append(record)
    samples = {s: metrics(cm) for s, cm in cms.items()}
    values = [v['macro_tissue']['dice'] for v in samples.values() if v['macro_tissue']['dice'] is not None]
    report = {
        'method': artifact['method'],
        'split': split,
        'model_sha256': digest(model_path),
        'manifest_sha256': digest(manifest),
        'label_ids': LABELS,
        'sections': sections,
        'samples': samples,
        'pooled': metrics(pooled) if cms else None,
        'mean_sample_macro_dice': float(np.mean(values)) if values else None,
        'note': 'Undefined metrics are null. Tissue macro excludes background. Unannotated pixels (255) are ignored.',
    }
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
