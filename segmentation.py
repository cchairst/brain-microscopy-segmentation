"""Reproducible 2D microscopy baselines. Run `python segmentation.py --help`."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import importlib.metadata

import joblib
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, uniform_filter
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

LABELS = {0: 'background', 1: 'gray_matter', 2: 'white_matter', 3: 'fluid_space'}
IGNORE = 255  # unannotated, uncertain tissue, or excluded artifact


def read_array(path, key='', channel=''):
    path = Path(path)
    if path.suffix.lower() == '.npy':
        a = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == '.mat':
        if not key:
            raise ValueError('MAT input requires an explicit mat_key in the manifest.')
        try:
            d = loadmat(path)
        except NotImplementedError as e:
            raise ValueError('MAT v7.3 is not supported; export a numeric 2D array to NPY or MAT v7.2.') from e
        if key not in d:
            raise ValueError(f'MAT key {key!r} missing from {path}.')
        a = np.asarray(d[key])
    else:
        a = np.asarray(Image.open(path))
    if channel != '':
        if a.ndim != 3:
            raise ValueError('Channel selection expects a channel-last H x W x C array.')
        a = a[..., int(channel)]
    if a.ndim != 2 or min(a.shape) < 2:
        raise ValueError(f'{path}: expected a 2D array; got {a.shape}. Select a numeric channel explicitly for RGB.')
    if not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all():
        raise ValueError(f'{path}: data must be finite and numeric.')
    return a


def read_mask(path, shape):
    a = read_array(path)
    if a.shape != shape or not np.isin(a, list(LABELS) + [IGNORE]).all():
        raise ValueError('Mask must match the image shape and contain only IDs 0,1,2,3,255.')
    return a.astype(np.uint8)


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        if not {'sample_id', 'section_id', 'split', 'image', 'mask'} <= set(reader.fieldnames or []):
            raise ValueError('Manifest requires sample_id,section_id,split,image,mask columns.')
        rows = list(reader)
    if not rows:
        raise ValueError('Manifest is empty.')
    groups, seen, images = {}, set(), set()
    for r in rows:
        if not r['sample_id'] or not r['section_id'] or r['split'] not in {'train', 'val', 'test'}:
            raise ValueError('Each row needs sample/section IDs and a train, val, or test split.')
        previous = groups.setdefault(r['sample_id'], r['split'])
        if previous != r['split']:
            raise ValueError(f"Leakage: sample {r['sample_id']} occurs in multiple splits.")
        identity = (r['sample_id'], r['section_id'])
        if identity in seen:
            raise ValueError('Duplicate sample/section row.')
        seen.add(identity)
        for name in ['image', 'mask']:
            if r[name]:
                r[name] = str((path.parent / r[name]).resolve())
        if not r['image'] or r['image'] in images:
            raise ValueError('Every row must reference a distinct image path.')
        images.add(r['image'])
    return rows


def normalize(a):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / max(float(hi - lo), 1e-6), 0, 1)


def features(a, method):
    x = normalize(a)
    if method == 'kmeans':
        return x[..., None]
    mean = uniform_filter(x, size=5)
    std = np.sqrt(np.maximum(uniform_filter(x * x, size=5) - mean * mean, 0))
    gy, gx = np.gradient(gaussian_filter(x, 1))
    return np.stack([x, gaussian_filter(x, 1), gaussian_filter(x, 3), mean, std, np.hypot(gx, gy)], -1)


def load_row(r):
    return read_array(r['image'], r.get('mat_key', ''), r.get('channel', ''))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def train(manifest, method, output, seed=42, pixels=20000):
    if pixels < 4:
        raise ValueError('pixels must be at least 4.')
    rows = read_manifest(manifest)
    train_rows = [r for r in rows if r['split'] == 'train']
    if not train_rows:
        raise ValueError('No training rows.')
    rng = np.random.default_rng(seed)
    xs, ys, provenance = [], [], []
    for r in train_rows:
        if not r['mask']:
            raise ValueError('Training requires expert masks; clustering uses them to map clusters to tissue IDs.')
        a = load_row(r)
        y = read_mask(r['mask'], a.shape).ravel()
        f = features(a, method).reshape(-1, 1 if method == 'kmeans' else 6)
        # Balanced sampling per section; ignore unannotated/artifact pixels.
        ids = []
        for label in LABELS:
            candidates = np.flatnonzero(y == label)
            if len(candidates):
                ids.extend(rng.choice(candidates, min(len(candidates), pixels // 4), replace=False))
        if not ids:
            raise ValueError('Training section contains no labeled pixels.')
        xs.append(f[ids]); ys.append(y[ids])
        provenance.append({'sample_id': r['sample_id'], 'section_id': r['section_id'],
                           'image_sha256': digest(r['image']), 'mask_sha256': digest(r['mask'])})
    x, y = np.concatenate(xs), np.concatenate(ys)
    if not {1, 2, 3} <= set(np.unique(y)):
        raise ValueError('Training annotations must include all three target tissue classes.')
    mapping = None
    if method == 'kmeans':
        model = MiniBatchKMeans(n_clusters=4, random_state=seed, n_init=3, batch_size=1024)
        model.fit(x)
        clusters = model.predict(x)
        mapping = []
        for k in range(4):
            members = y[clusters == k]
            mapping.append(int(np.bincount(members, minlength=4).argmax()) if len(members) else 0)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=2,
                                       class_weight='balanced', n_jobs=-1, random_state=seed)
        model.fit(x, y)
    artifact = {'model': model, 'method': method, 'mapping': mapping, 'labels': LABELS,
                'train_sample_ids': sorted({r['sample_id'] for r in train_rows}),
                'provenance': provenance, 'manifest_sha256': digest(manifest),
                'seed': seed, 'pixels_per_section': pixels, 'python': platform.python_version(),
                'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'scikit-learn', 'joblib']}}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    return artifact


def predict_array(artifact, a):
    f = features(a, artifact['method'])
    flat = f.reshape(-1, f.shape[-1])
    # Chunk classifier prediction to limit transient prediction allocations.
    pred = np.concatenate([artifact['model'].predict(flat[i:i+65536]) for i in range(0, len(flat), 65536)])
    if artifact['mapping'] is not None:
        pred = np.asarray(artifact['mapping'])[pred]
    return pred.reshape(a.shape).astype(np.uint8)


def metrics(cm):
    records = {}
    for c, name in LABELS.items():
        tp = int(cm[c, c]); actual = int(cm[c].sum()); predicted = int(cm[:, c].sum())
        def ratio(n, d):
            return float(n / d) if d else None
        records[name] = {'support': actual, 'dice': ratio(2 * tp, actual + predicted),
                         'iou': ratio(tp, actual + predicted - tp),
                         'precision': ratio(tp, predicted), 'recall': ratio(tp, actual)}
    macro = {}
    for key in ['dice', 'iou', 'precision', 'recall']:
        vals = [records[LABELS[c]][key] for c in [1, 2, 3] if records[LABELS[c]][key] is not None]
        macro[key] = float(np.mean(vals)) if vals else None
    return {'per_class': records, 'macro_tissue': macro, 'confusion_matrix': cm.tolist()}


def run_predictions(manifest, model_path, split, output):
    rows = read_manifest(manifest)
    artifact = joblib.load(model_path)  # Only load trusted artifacts; joblib can execute code.
    selected = [r for r in rows if r['split'] == split]
    if not selected:
        raise ValueError(f'No {split} rows.')
    if split != 'train' and {r['sample_id'] for r in selected} & set(artifact['train_sample_ids']):
        raise ValueError('Evaluation samples overlap model training samples.')
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    cms, sections, pooled = {}, [], np.zeros((4, 4), dtype=np.int64)
    for i, r in enumerate(selected):
        a = load_row(r); prediction = predict_array(artifact, a)
        filename = f'{i:05d}_labels.png'
        Image.fromarray(prediction).save(out / filename)
        record = {'sample_id': r['sample_id'], 'section_id': r['section_id'], 'prediction': filename,
                  'image_sha256': digest(r['image']), 'annotated': bool(r['mask'])}
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
    report = {'method': artifact['method'], 'split': split, 'model_sha256': digest(model_path),
              'manifest_sha256': digest(manifest), 'label_ids': LABELS,
              'sections': sections, 'samples': samples,
              'pooled': metrics(pooled) if cms else None,
              'mean_sample_macro_dice': float(np.mean(values)) if values else None,
              'note': 'Undefined metrics are null. Tissue macro excludes background. Unannotated pixels (255) are ignored.'}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def demo(root):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[:96, :96]
    rows = []
    for i, split in enumerate(['train', 'train', 'train', 'val', 'test', 'test']):
        radius = np.hypot(xx - 48 - rng.uniform(-3, 3), yy - 48)
        mask = np.zeros((96, 96), dtype=np.uint8)
        mask[radius < 40] = 1; mask[radius < 28] = 2; mask[radius < 10] = 3
        image = np.asarray([0.05, 0.4, 0.8, 0.15])[mask] + rng.normal(0, 0.025, mask.shape)
        np.save(root / f'brain{i}.npy', image.astype(np.float32))
        Image.fromarray(mask).save(root / f'brain{i}_mask.png')
        rows.append({'sample_id': f'synthetic_brain{i}', 'section_id': '001', 'split': split,
                     'image': f'brain{i}.npy', 'mask': f'brain{i}_mask.png'})
    manifest = root / 'manifest.csv'
    with manifest.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for method in ['kmeans', 'rf']:
        path = root / f'{method}.joblib'
        train(manifest, method, path, pixels=4000)
        report = run_predictions(manifest, path, 'test', root / method)
        print(f"SYNTHETIC DEMO ONLY {method}: mean sample Dice = {report['mean_sample_macro_dice']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train'); p.add_argument('--manifest', required=True)
    p.add_argument('--method', choices=['kmeans', 'rf'], required=True); p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--pixels', type=int, default=20000)
    p = sub.add_parser('predict'); p.add_argument('--manifest', required=True); p.add_argument('--model', required=True)
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test'); p.add_argument('--output', required=True)
    p = sub.add_parser('demo'); p.add_argument('--output', default='outputs/synthetic_demo')
    args = parser.parse_args()
    if args.command == 'train':
        train(args.manifest, args.method, args.output, args.seed, args.pixels)
        print(f'Model saved to {args.output}')
    elif args.command == 'predict':
        report = run_predictions(args.manifest, args.model, args.split, args.output)
        print(f"Report saved to {args.output}/report.json; mean sample Dice: {report['mean_sample_macro_dice']}")
    else:
        demo(args.output)


if __name__ == '__main__':
    main()
