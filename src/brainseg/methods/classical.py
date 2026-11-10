"""Unsupervised classical segmentation: Otsu tissue mask, gray/white split, morphological cleanup."""

import numpy as np
from scipy.ndimage import binary_fill_holes, label

from brainseg.preprocess import normalize, otsu

METHODS = ('otsu', 'gmm', 'kmeans')


def remove_small(mask, min_size):
    """Drop connected components of `mask` smaller than `min_size` pixels."""
    components, _ = label(mask)
    keep = np.bincount(components.ravel()) >= min_size
    keep[0] = False
    return keep[components]


def segment(
    a,
    method='gmm',
    retardance=0,
    reflectivity=-1,
    features=None,
    pixel_size=0.055,
    min_area=0.2,
    min_fluid_area=1.2,
    seed=0,
    sample=200_000,
):
    """Label an H x W x C section as 0 background, 1 gray, 2 white, 3 fluid, without training labels.

    Tissue is Otsu on reflectivity over imaged pixels. Imaged holes enclosed by tissue are fluid when at least
    `min_fluid_area` mm^2, otherwise they are filled as tissue (vessels take the surrounding label). Gray and
    white are split on retardance (`otsu`) or by two clusters of the `features` contrasts (`gmm`, `kmeans`;
    default retardance only); white is the higher-retardance group. Islands smaller than `min_area` mm^2 are
    merged into their surroundings. Areas use `pixel_size` in mm. Clusters are fitted on at most `sample`
    tissue pixels. Damage (4) is not predicted; a section without tissue is all background.
    """
    if method not in METHODS:
        raise ValueError(f'unknown method: {method}')
    if method == 'otsu' and features is not None:
        raise ValueError('otsu splits on retardance only; features apply to gmm and kmeans.')
    cols = [retardance] if features is None else list(features)
    n = a.shape[-1]
    for c in {retardance, reflectivity, *cols}:
        if not -n <= c < n:
            raise ValueError(f'contrast index {c} is out of range for {n} images per row.')
    min_size = max(1, round(min_area / pixel_size**2))
    raw_ref = a[..., reflectivity]
    imaged = np.isfinite(raw_ref) & (np.nan_to_num(raw_ref) > 0)
    out = np.zeros(a.shape[:2], np.uint8)
    if not imaged.any():  # blank section: nothing to threshold (and no all-NaN warnings)
        return out
    ref = np.nan_to_num(normalize(raw_ref))
    tissue = remove_small(imaged & (ref > otsu(ref[imaged])), min_size)
    if not tissue.any():
        return out
    holes = binary_fill_holes(tissue) & ~tissue & imaged  # unimaged pixels stay background, as in predict
    fluid = remove_small(holes, max(1, round(min_fluid_area / pixel_size**2)))
    tissue |= holes & ~fluid
    # Scale the contrasts we use inside the tissue, so the gray/white split uses the tissue's own range.
    scaled = {c: np.nan_to_num(normalize(a[..., c], tissue)[tissue]) for c in {retardance, *cols}}
    ret = scaled[retardance]
    if method == 'otsu' or len(ret) < 2:
        is_white = ret > otsu(ret)
    else:
        values = np.stack([scaled[c] for c in cols], -1)
        rng = np.random.default_rng(seed)
        fit = values[rng.choice(len(values), min(sample, len(values)), replace=False)]
        if method == 'gmm':
            from sklearn.mixture import GaussianMixture  # deferred: keeps `brainseg --help` fast

            model = GaussianMixture(n_components=2, random_state=seed).fit(fit)
        else:
            from sklearn.cluster import KMeans

            model = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(fit)
        groups = model.predict(values)
        is_white = groups == np.argmax([ret[groups == k].mean() if (groups == k).any() else -1 for k in (0, 1)])
    white = np.zeros_like(tissue)
    white[tissue] = is_white
    white = remove_small(white, min_size)
    gray = remove_small(tissue & ~white, min_size)
    out[tissue] = 2
    out[gray] = 1
    out[fluid] = 3
    return out
