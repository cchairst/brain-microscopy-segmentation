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


@pytest.mark.parametrize('position', [(1, 1), (7, 13), (63, 63)])
def test_fit_uses_sparse_labels_between_grid_points(position):
    a, y = section(0)
    labels = np.full_like(y, IGNORE)
    # A single imaged reference pixel is enough to centre a valid training patch.
    a[position] = [0.4, 0.8]
    labels[position] = 1
    model, losses = fit([a], [labels], iterations=2, patch=16, batch=2, width=2, depth=1, seed=5, on='cpu')
    assert len(losses) == 2 and np.isfinite(losses).all()
    assert predict(model, a, on='cpu').shape == labels.shape


@pytest.mark.parametrize('value', [0, np.nan])
def test_sparse_labels_on_unimaged_pixels_are_not_used(value):
    a, y = section(0)
    labels = np.full_like(y, IGNORE)
    labels[7, 13] = 1
    a[7, 13] = value
    with pytest.raises(ValueError, match='No labelled'):
        fit([a], [labels], iterations=2, patch=16, batch=2, width=2, depth=1, on='cpu')


def test_sparse_label_training_is_reproducible():
    a, y = section(0)
    labels = np.full_like(y, IGNORE)
    labels[7, 13] = 1
    labels[51, 53] = 2
    # An empty section is skipped without losing either labelled location in the other section.
    args = ([a, a], [np.full_like(y, IGNORE), labels])
    settings = dict(iterations=4, patch=16, batch=2, width=2, depth=1, seed=11, on='cpu')
    first, losses = fit(*args, **settings)
    second, repeated = fit(*args, **settings)
    np.testing.assert_array_equal(losses, repeated)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
