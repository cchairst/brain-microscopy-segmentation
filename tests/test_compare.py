import csv
import json

import numpy as np
import pytest

from brainseg.benchmark import demo, run_predictions
from brainseg.cli import main
from brainseg.compare import bootstrap_ci, compare
from brainseg.metrics import hd95


def test_hd95_spacing_scales_distances():
    a = np.zeros((40, 40), int)
    b = np.zeros((40, 40), int)
    a[10:20, 10:20] = 1
    b[13:23, 10:20] = 1
    assert hd95(a, b, 1) == 3.0
    assert hd95(a, b, 1, spacing=55.0) == pytest.approx(165.0)


def test_report_gives_hd95_in_um_only_with_pixel_size(tmp_path):
    demo(tmp_path)
    with (tmp_path / 'manifest.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['pixel_size_mm'] = '0.055' if r['sample_id'] == 'synthetic_brain4' else ''
    with (tmp_path / 'manifest.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report = run_predictions(tmp_path / 'manifest.csv', tmp_path / 'kmeans.joblib', 'test', tmp_path / 'out')
    with_size, without = report['sections']
    for name, px in with_size['metrics']['hd95_px'].items():
        um = with_size['metrics']['hd95_um'][name]
        assert (um is None) if px is None else um == pytest.approx(55 * px)
    assert all(v is None for v in without['metrics']['hd95_um'].values())
    assert report['samples']['synthetic_brain4']['hd95_um'] == with_size['metrics']['hd95_um']


def test_bootstrap_ci():
    assert bootstrap_ci([0.7]) == (0.7, None, None)
    values = [0.6, 0.7, 0.8, 0.9]
    mean, low, high = bootstrap_ci(values)
    assert mean == pytest.approx(0.75) and 0.6 <= low < mean < high <= 0.9
    spread = list(np.linspace(0.5, 0.95, 10))
    assert bootstrap_ci(spread, n_boot=200) == bootstrap_ci(spread, n_boot=200)  # seeded, so reproducible
    assert bootstrap_ci(spread, n_boot=200, seed=1) != bootstrap_ci(spread, n_boot=200)


def fake_report(path, method, dice_by_section):
    """A minimal report: one annotated section per (sample, macro Dice) pair, plus one unannotated section."""
    sections = [{'sample_id': 'brainA', 'section_id': 'x', 'annotated': False}]
    for i, (sample, dice) in enumerate(dice_by_section):
        sections.append(
            {
                'sample_id': sample,
                'section_id': str(i),
                'metrics': {
                    'macro_tissue': {'dice': dice},
                    'per_class': {'gray_matter': {'dice': dice, 'iou': None}},
                    'hd95_px': {'gray_matter': 2.0},
                    'hd95_um': {'gray_matter': None},
                    'volume_fraction': {
                        'reference': {'gray_matter': 0.5, 'fluid_space': None},
                        'prediction': {'gray_matter': 0.6, 'fluid_space': 0.1},  # over-estimate: error is |0.6 - 0.5|
                    },
                },
            }
        )
    path.mkdir(parents=True)
    (path / 'report.json').write_text(json.dumps({'method': method, 'sections': sections}))
    return path


def test_compare_summarizes_per_brain_and_skips_undefined(tmp_path):
    a = fake_report(tmp_path / 'gmm_test', 'gmm', [('brainA', 0.8), ('brainA', 0.6), ('brainB', 0.9)])
    b = fake_report(tmp_path / 'rf_test', 'rf', [('brainA', 0.5)])
    rows = compare([a, b / 'report.json'], tmp_path / 'results' / 'table.csv')
    by_key = {(r['run'], r['sample_id'], r['class'], r['metric']): r for r in rows}
    gmm_a = by_key['gmm_test', 'brainA', 'macro_tissue', 'dice']
    assert gmm_a['n'] == 2 and gmm_a['mean'] == pytest.approx(0.7) and gmm_a['ci_low'] is not None
    assert by_key['gmm_test', 'brainB', 'macro_tissue', 'dice']['ci_low'] is None  # one section: no interval
    assert by_key['rf_test', 'brainA', 'gray_matter', 'dice']['method'] == 'rf'
    assert by_key['gmm_test', 'brainA', 'gray_matter', 'vf_error']['mean'] == pytest.approx(0.1)
    assert by_key['gmm_test', 'brainA', 'gray_matter', 'hd95_px']['mean'] == 2.0
    # Undefined values (null IoU, null HD95 in um, missing reference fraction) produce no rows.
    assert not any(k[3] in ('iou', 'hd95_um') or k[2] == 'fluid_space' for k in by_key)
    with (tmp_path / 'results' / 'table.csv').open(newline='') as f:
        assert len(list(csv.DictReader(f))) == len(rows)


def test_compare_rejects_duplicate_run_names(tmp_path):
    a = fake_report(tmp_path / 'one' / 'test', 'gmm', [('brainA', 0.8)])
    b = fake_report(tmp_path / 'two' / 'test', 'rf', [('brainA', 0.8)])
    with pytest.raises(ValueError, match='run name'):
        compare([a, b], tmp_path / 'table.csv')


def test_compare_command(tmp_path, capsys):
    demo(tmp_path)
    main(['compare', str(tmp_path / 'kmeans'), str(tmp_path / 'rf'), '--output', str(tmp_path / 'table.csv')])
    out = capsys.readouterr().out
    assert 'tissue Dice' in out and 'rows written' in out
    with (tmp_path / 'table.csv').open(newline='') as f:
        assert {r['run'] for r in csv.DictReader(f)} == {'kmeans', 'rf'}
