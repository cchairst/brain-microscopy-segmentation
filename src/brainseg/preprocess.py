"""Intensity scaling shared by all methods."""

import numpy as np


def percentile_bounds(a, mask=None):
    """1st and 99th percentiles of `a`, inside `mask` if given."""
    return np.percentile(a if mask is None else a[mask], [1, 99])


def scale(a, lo, hi):
    """Clip to [lo, hi] and scale to [0, 1]."""
    return np.clip((a - lo) / max(float(hi - lo), 1e-6), 0, 1)


def normalize(a, mask=None):
    """Clip to the 1st/99th percentiles (inside `mask` if given) and scale to [0, 1]."""
    a = a.astype(np.float32)
    return scale(a, *percentile_bounds(a, mask))
