import numpy as np
import pytest

torch = pytest.importorskip('torch')

from brainseg import IGNORE  # noqa: E402
from brainseg.methods.unet import fit, load, predict, save  # noqa: E402


def section(seed):
    """Two-contrast section: gray disk with a white core in gel, an unimaged border; and its labels."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:64, :64]
    r = np.hypot(yy - 32 - rng.uniform(-4, 4), xx - 32 - rng.uniform(-4, 4))
    labels = np.select([r < 10, r < 22], [2, 1], 0).astype(np.uint8)
    ret = np.choose(labels, [0.1, 0.3, 0.9]) + rng.normal(0, 0.03, r.shape)
    ref = np.choose(labels, [0.3, 0.9, 0.9]) + rng.normal(0, 0.03, r.shape)
    a = np.stack([ret, ref], -1).astype(np.float32)
    a[:, :4] = 0  # unimaged
    return a, labels


def test_unet_learns_and_ignores_unlabelled_pixels(tmp_path):
    train = [section(i) for i in range(4)]
    labels = [y.copy() for _, y in train]
    for y in labels:
        y[:, 40:] = IGNORE  # half of each section unlabelled
    model, losses = fit([a for a, _ in train], labels, iterations=150, patch=32, batch=8, width=8, depth=2, on='cpu')
    assert np.mean(losses[-20:]) < np.mean(losses[:20])
    a, truth = section(9)
    pred = predict(model, a, on='cpu')
    assert pred.shape == truth.shape and (pred[:, :4] == 0).all()
    assert (pred[:, 4:] == truth[:, 4:]).mean() > 0.9
    save(model, tmp_path / 'm.pt', train_sample_ids=['s'])
    loaded, meta = load(tmp_path / 'm.pt')
    assert meta == {'train_sample_ids': ['s']}
    np.testing.assert_array_equal(predict(loaded, a, on='cpu'), pred)


def test_fit_rejects_sections_without_labels():
    a, y = section(0)
    with pytest.raises(ValueError, match='No labelled'):
        fit([a], [np.full_like(y, IGNORE)], iterations=1, on='cpu')
