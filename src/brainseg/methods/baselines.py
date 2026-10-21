"""Pixel classifiers trained on annotated sections: k-means mapped to tissue IDs, and a random forest."""

import importlib.metadata
import platform
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier

from brainseg import LABELS
from brainseg.io import digest, load_row, read_manifest, read_mask
from brainseg.preprocess import normalize


def features(a, method):
    x = normalize(a)
    if method == 'kmeans':
        return x[..., None]
    mean = uniform_filter(x, size=5)
    std = np.sqrt(np.maximum(uniform_filter(x * x, size=5) - mean * mean, 0))
    gy, gx = np.gradient(gaussian_filter(x, 1))
    return np.stack([x, gaussian_filter(x, 1), gaussian_filter(x, 3), mean, std, np.hypot(gx, gy)], -1)


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
        xs.append(f[ids])
        ys.append(y[ids])
        provenance.append(
            {
                'sample_id': r['sample_id'],
                'section_id': r['section_id'],
                'image_sha256': digest(r['image']),
                'mask_sha256': digest(r['mask']),
            }
        )
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
        model = RandomForestClassifier(
            n_estimators=100, max_depth=20, min_samples_leaf=2, class_weight='balanced', n_jobs=-1, random_state=seed
        )
        model.fit(x, y)
    artifact = {
        'model': model,
        'method': method,
        'mapping': mapping,
        'labels': LABELS,
        'train_sample_ids': sorted({r['sample_id'] for r in train_rows}),
        'provenance': provenance,
        'manifest_sha256': digest(manifest),
        'seed': seed,
        'pixels_per_section': pixels,
        'python': platform.python_version(),
        'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'scikit-learn', 'joblib']},
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    return artifact


def predict_array(artifact, a):
    f = features(a, artifact['method'])
    flat = f.reshape(-1, f.shape[-1])
    # Chunk classifier prediction to limit transient prediction allocations.
    pred = np.concatenate([artifact['model'].predict(flat[i : i + 65536]) for i in range(0, len(flat), 65536)])
    if artifact['mapping'] is not None:
        pred = np.asarray(artifact['mapping'])[pred]
    return pred.reshape(a.shape).astype(np.uint8)
