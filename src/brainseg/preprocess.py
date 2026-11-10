"""Intensity scaling shared by all methods."""

import numpy as np


def otsu(values):
    """Otsu threshold of `values` from a 256-bin histogram."""
    h, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w = np.cumsum(h).astype(float)
    s = np.cumsum(h * centers)
    denominator = w * (w[-1] - w)
    score = np.divide((s[-1] * w - s * w[-1]) ** 2, denominator, out=np.zeros_like(w), where=denominator > 0)
    return float(centers[np.argmax(score)])


def percentile_bounds(a, mask=None):
    """1st and 99th percentiles of `a`, inside `mask` if given. NaNs are ignored."""
    return np.nanpercentile(a if mask is None else a[mask], [1, 99])


def scale(a, lo, hi):
    """Clip to [lo, hi] and scale to [0, 1]."""
    return np.clip((a - lo) / max(float(hi - lo), 1e-6), 0, 1)


def orientation(z):
    """In-plane fiber angle in radians, [-pi/2, pi/2], from the complex optic-axis value (TEnAO).

    The pipeline averages in the doubled-angle domain, so the angle is half the phase.
    """
    return np.angle(z) / 2


def normalize(a, mask=None):
    """Clip to the 1st/99th percentiles (inside `mask` if given) and scale to [0, 1]. NaNs are kept."""
    a = a.astype(np.float32)
    return scale(a, *percentile_bounds(a, mask))
