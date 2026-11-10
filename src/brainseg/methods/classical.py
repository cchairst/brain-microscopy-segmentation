"""Unsupervised classical segmentation: Otsu tissue mask, gray/white split, morphological cleanup."""

import numpy as np
from scipy.ndimage import binary_fill_holes, label
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from brainseg.preprocess import normalize, otsu

METHODS = ('otsu', 'gmm', 'kmeans')


def remove_small(mask, min_size):
    """Drop connected components of `mask` smaller than `min_size` pixels."""
    components, _ = label(mask)
    keep = np.bincount(components.ravel()) >= min_size
    keep[0] = False
    return keep[components]


def segment(
    a, method='gmm', retardance=0, reflectivity=-1, features=None, min_size=64, min_fluid=400, seed=0, sample=200_000
):
    """Label an H x W x C section as 0 background, 1 gray, 2 white, 3 fluid, without training labels.

    Tissue is Otsu on reflectivity. Holes enclosed by tissue are fluid when at least `min_fluid` pixels,
    otherwise they are filled as tissue (vessels take the surrounding label). Gray and white are split on
    retardance (`otsu`) or by two clusters of the `features` contrasts (`gmm`, `kmeans`; default retardance
    only); white is the higher-retardance group. Components smaller than `min_size` pixels are merged into
    their surroundings.
    Clusters are fitted on at most `sample` tissue pixels. Damage (4) is not predicted.
    """
    if method not in METHODS:
        raise ValueError(f'unknown method: {method}')
    raw_ref = a[..., reflectivity]
    imaged = np.isfinite(raw_ref) & (np.nan_to_num(raw_ref) > 0)
    ref = np.nan_to_num(normalize(raw_ref))
    tissue = remove_small(imaged & (ref > otsu(ref[imaged])), min_size)
    holes = binary_fill_holes(tissue) & ~tissue
    fluid = remove_small(holes, min_fluid)
    tissue |= holes & ~fluid
    # Scale each contrast inside the tissue, so the gray/white split uses the tissue's own range.
    x = np.nan_to_num(np.stack([normalize(a[..., c], tissue) for c in range(a.shape[-1])], -1))
    ret = x[..., retardance][tissue]
    if method == 'otsu':
        is_white = ret > otsu(ret)
    else:
        values = x[tissue][:, [retardance] if features is None else list(features)]
        rng = np.random.default_rng(seed)
        fit = values[rng.choice(len(values), min(sample, len(values)), replace=False)]
        if method == 'gmm':
            model = GaussianMixture(n_components=2, random_state=seed).fit(fit)
        else:
            model = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(fit)
        groups = model.predict(values)
        is_white = groups == np.argmax([ret[groups == k].mean() if (groups == k).any() else -1 for k in (0, 1)])
    white = np.zeros_like(tissue)
    white[tissue] = is_white
    white = remove_small(white, min_size)
    gray = remove_small(tissue & ~white, min_size)
    out = np.zeros(a.shape[:2], np.uint8)
    out[tissue] = 2
    out[gray] = 1
    out[fluid] = 3
    return out
