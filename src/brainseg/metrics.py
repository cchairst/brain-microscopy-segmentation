"""Benchmark metrics from a confusion matrix."""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from brainseg import LABELS


def metrics(cm):
    records = {}
    for c, name in LABELS.items():
        tp = int(cm[c, c])
        actual = int(cm[c].sum())
        predicted = int(cm[:, c].sum())

        def ratio(n, d):
            return float(n / d) if d else None

        records[name] = {
            'support': actual,
            'dice': ratio(2 * tp, actual + predicted),
            'iou': ratio(tp, actual + predicted - tp),
            'precision': ratio(tp, predicted),
            'recall': ratio(tp, actual),
        }
    macro = {}
    for key in ['dice', 'iou', 'precision', 'recall']:
        vals = [records[LABELS[c]][key] for c in [1, 2, 3] if records[LABELS[c]][key] is not None]
        macro[key] = float(np.mean(vals)) if vals else None
    return {
        'per_class': records,
        'macro_tissue': macro,
        'confusion_matrix': cm.tolist(),
        'volume_fraction': {'reference': tissue_fractions(cm.sum(1)), 'prediction': tissue_fractions(cm.sum(0))},
    }


def tissue_fractions(counts):
    """Fraction of tissue pixels (classes 1-3) in each tissue class."""
    total = counts[1:4].sum()
    return {LABELS[c]: float(counts[c] / total) if total else None for c in [1, 2, 3]}


def hd95(pred, ref, label, valid=None):
    """95th-percentile symmetric surface distance in pixels within `valid`; None if either side is empty."""
    valid = np.ones(ref.shape, bool) if valid is None else valid
    p, r = (pred == label) & valid, (ref == label) & valid
    if not p.any() or not r.any():
        return None
    sp, sr = p & ~binary_erosion(p), r & ~binary_erosion(r)
    return float(np.percentile(np.concatenate([distance_transform_edt(~sr)[sp], distance_transform_edt(~sp)[sr]]), 95))
