import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.io import savemat

from brainseg.benchmark import demo, run_predictions
from brainseg.io import read_array, read_manifest, read_mask
from brainseg.methods.baselines import train
from brainseg.metrics import hd95, metrics
from brainseg.preprocess import normalize, orientation


def test_exact_metrics_and_absent_classes():
    cm = np.zeros((5, 5), int)
    cm[:4, :4] = [[5, 0, 0, 0], [0, 3, 1, 0], [0, 1, 3, 0], [0, 0, 0, 0]]
    m = metrics(cm)
    assert m['per_class']['gray_matter']['dice'] == pytest.approx(0.75)
    assert m['per_class']['white_matter']['iou'] == pytest.approx(0.6)
    assert m['per_class']['fluid_space']['dice'] is None
    assert m['macro_tissue']['dice'] == pytest.approx(0.75)


def test_background_false_positives_count():
    cm = np.zeros((5, 5), int)
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


def test_hd95_volume_fractions_and_nan_normalize():
    a = np.zeros((40, 40), int)
    b = np.zeros((40, 40), int)
    a[10:20, 10:20] = 1
    b[13:23, 10:20] = 1
    assert hd95(a, b, 1) == 3.0
    assert hd95(a, b, 2) is None
    cm = np.zeros((5, 5), int)
    cm[1, 1] = 2
    cm[2, 2] = 1
    cm[3, 2] = 1
    vf = metrics(cm)['volume_fraction']
    assert vf['reference'] == {'gray_matter': 0.5, 'white_matter': 0.25, 'fluid_space': 0.25}
    assert vf['prediction']['white_matter'] == 0.5
    out = normalize(np.array([np.nan, 0, 1, 2]))
    assert np.isnan(out[0])
    assert np.nanmin(out) == pytest.approx(0) and np.nanmax(out) == pytest.approx(1)


def test_multicontrast_rows(tmp_path):
    demo(tmp_path)
    with (tmp_path / 'manifest.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:  # second contrast: an inverted copy of the first
        second = r['image'].replace('.npy', '_inv.npy')
        np.save(tmp_path / second, 1 - np.load(tmp_path / r['image']))
        r['image'] = f'{r["image"]};{second}'
    write_manifest(tmp_path / 'multi.csv', rows)
    artifact = train(tmp_path / 'multi.csv', 'rf', tmp_path / 'multi.joblib', pixels=4000)
    assert artifact['contrasts'] == 2
    report = run_predictions(tmp_path / 'multi.csv', tmp_path / 'multi.joblib', 'test', tmp_path / 'multi_out')
    assert report['mean_sample_macro_dice'] > 0.9
    section = report['sections'][0]
    assert len(section['image_sha256']) == 2
    assert section['metrics']['hd95_px']['white_matter'] is not None
    # A single-contrast model must reject two-contrast input.
    with pytest.raises(ValueError):
        run_predictions(tmp_path / 'multi.csv', tmp_path / 'rf.joblib', 'test', tmp_path / 'bad')


def test_raw_section_values(tmp_path):
    p = tmp_path / 'raw.npy'
    np.save(p, np.array([[np.nan, 1.0], [2.0, 3.0]]))
    assert np.isnan(read_array(p)[0, 0])
    np.save(p, np.array([[np.inf, 1.0], [2.0, 3.0]]))
    with pytest.raises(ValueError):
        read_array(p)
    np.save(p, np.ones((2, 2), complex))
    with pytest.raises(ValueError, match='orientation'):
        read_array(p)
    theta = np.array([-1.5, -0.3, 0.0, 0.7, 1.5])
    np.testing.assert_allclose(orientation(3.0 * np.exp(2j * theta)), theta)


def test_nan_pixels_are_ignored_and_predicted_background(tmp_path):
    demo(tmp_path)
    for name in ['brain0.npy', 'brain4.npy']:  # one train and one test section lose their left edge
        a = np.load(tmp_path / name)
        a[:, :20] = np.nan
        np.save(tmp_path / name, a)
    train(tmp_path / 'manifest.csv', 'rf', tmp_path / 'nan.joblib', pixels=4000)
    report = run_predictions(tmp_path / 'manifest.csv', tmp_path / 'nan.joblib', 'test', tmp_path / 'nan_out')
    assert report['mean_sample_macro_dice'] > 0.9
    first = report['sections'][0]
    assert first['sample_id'] == 'synthetic_brain4'
    prediction = np.asarray(Image.open(tmp_path / 'nan_out' / first['prediction']))
    assert (prediction[:, :20] == 0).all()


def test_damage_is_scored_but_not_tissue():
    cm = np.zeros((5, 5), int)
    cm[1, 1] = 4
    cm[2, 2] = 4
    cm[3, 3] = 2
    cm[4, 4] = 3
    cm[4, 1] = 1  # one damaged pixel predicted as gray matter
    m = metrics(cm)
    assert m['per_class']['damage']['dice'] == pytest.approx(6 / 7)
    assert m['per_class']['damage']['recall'] == pytest.approx(0.75)
    assert m['macro_tissue']['precision'] == pytest.approx((0.8 + 1 + 1) / 3)  # damage lowers GM precision only
    assert m['volume_fraction']['reference'] == {'gray_matter': 0.4, 'white_matter': 0.4, 'fluid_space': 0.2}
    pred = np.zeros((40, 40), int)
    ref = np.zeros((40, 40), int)
    pred[10:20, 10:20] = 4
    ref[13:23, 10:20] = 4
    assert hd95(pred, ref, 4) == 3.0


def test_damage_masks_train_and_predict(tmp_path):
    demo(tmp_path)
    for i in [0, 4]:  # a damaged patch in one train and one test reference
        a = np.load(tmp_path / f'brain{i}.npy')
        a[60:70, 40:56] = 0.6
        np.save(tmp_path / f'brain{i}.npy', a)
        mask = np.asarray(Image.open(tmp_path / f'brain{i}_mask.png')).copy()
        mask[60:70, 40:56] = 4
        Image.fromarray(mask).save(tmp_path / f'brain{i}_mask.png')
    train(tmp_path / 'manifest.csv', 'rf', tmp_path / 'dmg.joblib', pixels=5000)
    report = run_predictions(tmp_path / 'manifest.csv', tmp_path / 'dmg.joblib', 'test', tmp_path / 'dmg_out')
    first = report['sections'][0]['metrics']
    assert len(first['confusion_matrix']) == 5
    assert first['per_class']['damage']['support'] == 160
    assert 'damage' in first['hd95_px'] and 'damage' not in first['volume_fraction']['reference']
