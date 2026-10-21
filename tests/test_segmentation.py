import csv
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from brainseg.benchmark import demo, run_predictions
from brainseg.io import read_array, read_manifest, read_mask
from brainseg.metrics import metrics
from brainseg.preprocess import normalize


def test_exact_metrics_and_absent_classes():
    cm = np.array([[5, 0, 0, 0], [0, 3, 1, 0], [0, 1, 3, 0], [0, 0, 0, 0]])
    m = metrics(cm)
    assert m['per_class']['gray_matter']['dice'] == pytest.approx(0.75)
    assert m['per_class']['white_matter']['iou'] == pytest.approx(0.6)
    assert m['per_class']['fluid_space']['dice'] is None
    assert m['macro_tissue']['dice'] == pytest.approx(0.75)


def test_background_false_positives_count():
    cm = np.zeros((4, 4), int)
    cm[0, 1] = 2
    cm[1, 1] = 2
    assert metrics(cm)['per_class']['gray_matter']['precision'] == pytest.approx(0.5)


def test_io_and_mask_validation(tmp_path):
    p = tmp_path / 'section.mat'
    a = np.arange(12).reshape(3, 4)
    savemat(p, {'intensity': a})
    np.testing.assert_array_equal(read_array(p, 'intensity'), a)
    with pytest.raises(ValueError):
        read_array(p)
    p = tmp_path / 'mask.npy'
    np.save(p, np.full((3, 4), 255))
    assert (read_mask(p, (3, 4)) == 255).all()
    np.save(p, np.full((3, 4), 8))
    with pytest.raises(ValueError):
        read_mask(p, (3, 4))
    np.testing.assert_array_equal(normalize(np.ones((3, 4))), 0)


def write_manifest(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_end_to_end_and_leakage(tmp_path):
    demo(tmp_path)
    manifest = tmp_path / 'manifest.csv'
    read_manifest(manifest)
    for method in ['kmeans', 'rf']:
        report = run_predictions(manifest, tmp_path / f'{method}.joblib', 'test', tmp_path / 'check' / method)
        assert len(report['samples']) == 2
        assert report['mean_sample_macro_dice'] > 0.9
    # Separate manifest still cannot evaluate a trained sample as a test sample.
    with manifest.open(newline='') as f:
        original = list(csv.DictReader(f))
    write_manifest(tmp_path / 'leaked.csv', [dict(original[0], split='test')])
    with pytest.raises(ValueError):
        run_predictions(tmp_path / 'leaked.csv', tmp_path / 'rf.joblib', 'test', tmp_path / 'bad')
    original[-1]['sample_id'] = original[0]['sample_id']
    write_manifest(manifest, original)
    with pytest.raises(ValueError):
        read_manifest(manifest)
