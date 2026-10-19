"""Benchmark metrics from a confusion matrix."""

import numpy as np

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
    return {'per_class': records, 'macro_tissue': macro, 'confusion_matrix': cm.tolist()}
