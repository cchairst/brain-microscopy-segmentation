import csv
import json

import numpy as np
import pytest
from PIL import Image

from brainseg.benchmark import run_classical
from brainseg.cli import main
from brainseg.methods.classical import remove_small, segment
from brainseg.preprocess import otsu

PIXELS = {'pixel_size': 1.0, 'min_area': 20, 'min_fluid_area': 100}  # areas in pixels for the 120 x 120 section
POINTS = {  # (row, col): expected label on the synthetic section
    (60, 60): 3,  # large hole in white matter: fluid
    (60, 75): 2,  # white matter
    (60, 103): 1,  # gray matter ring
    (60, 113): 0,  # gel inside the imaged area
    (2, 2): 0,  # outside the imaged area
    (15, 15): 0,  # tissue speck in the gel: removed
    (60, 90): 2,  # small hole in white matter: filled as white
    (18, 60): 1,  # small white island in gray matter: merged into gray
}


def expected_labels():
    """Hand-built reference for section(): what a correct segmentation should give."""
    yy, xx = np.mgrid[:120, :120]
    r = np.hypot(yy - 60, xx - 60)
    labels = np.select([r < 10, r < 35, r < 50], [3, 2, 1], 0).astype(np.uint8)
    labels[:5], labels[:, :5] = 0, 0
    return labels


def section(gel=0.3):
    """Ret, cross-pol, reflectivity stack: tissue disk in gel, white core, holes, specks."""
    yy, xx = np.mgrid[:120, :120]
    r = np.hypot(yy - 60, xx - 60)
    rng = np.random.default_rng(0)
    ref = np.where(r < 50, 0.9, gel) + rng.normal(0, 0.02, r.shape)
    ret = np.where(r < 35, 0.9, 0.3) + rng.normal(0, 0.02, r.shape)
    ref[r < 10] = gel  # ventricle-sized hole
    ref[59:61, 89:91] = gel  # vessel-sized hole
    ref[14:17, 14:17] = 0.9  # speck
    ret[17:20, 59:62] = 0.9  # white island in the gray ring (radius 42)
    ref[:5], ref[:, :5] = 0, 0  # outside the imaged area
    cro = rng.random(r.shape)
    return np.stack([ret, cro, ref], -1).astype(np.float32)


@pytest.mark.parametrize('method', ['otsu', 'gmm', 'kmeans'])
def test_segment_labels_each_structure(method):
    labels = segment(section(), method, **PIXELS)
    for (row, col), expected in POINTS.items():
        assert labels[row, col] == expected, (row, col)


def test_segment_unimaged_pixels_are_background():
    a = section()
    a[:, 100:, :] = np.nan
    labels = segment(a, 'otsu', **PIXELS)
    assert (labels[:, 100:] == 0).all() and labels[60, 75] == 2


def test_segment_otsu_ignores_unimaged_area():
    # Real gel is nearly as bright as tissue; a wide empty margin (outside the scanned tiles) must not drag the
    # tissue threshold down into it.
    a = np.pad(section(gel=0.75), ((120, 120), (120, 120), (0, 0)))
    labels = segment(a, 'otsu', **PIXELS)
    for (row, col), expected in POINTS.items():
        assert labels[row + 120, col + 120] == expected, (row, col)


@pytest.mark.filterwarnings('error')
@pytest.mark.parametrize('method', ['otsu', 'gmm', 'kmeans'])
def test_segment_without_tissue_is_background(method):
    assert (segment(np.zeros((50, 50, 3), np.float32), method, **PIXELS) == 0).all()
    assert (segment(np.full((50, 50, 3), np.nan, np.float32), method, **PIXELS) == 0).all()


def test_segment_with_imaged_pixels_but_no_tissue_left_is_background():
    # Tissue exists before cleanup but every island is below the minimum area.
    assert (segment(section(), 'gmm', pixel_size=1.0, min_area=1e6, min_fluid_area=100) == 0).all()


def test_segment_areas_are_physical():
    # The same physical thresholds give the same labels at a different pixel size; pixel counts would not.
    a = section()
    assert np.array_equal(
        segment(a, 'otsu', pixel_size=0.1, min_area=0.2, min_fluid_area=1.0), segment(a, 'otsu', **PIXELS)
    )


def test_segment_enclosed_unimaged_region_is_background():
    a = section()
    yy, xx = np.mgrid[:120, :120]
    a[np.hypot(yy - 60, xx - 60) < 10] = np.nan  # a missing tile where the fluid hole is
    labels = segment(a, 'otsu', **PIXELS)
    assert labels[60, 60] == 0 and labels[60, 75] == 2


def test_segment_rejects_bad_settings():
    a = section()
    with pytest.raises(ValueError, match='features apply'):
        segment(a, 'otsu', features=[0, 1])
    with pytest.raises(ValueError, match='out of range'):
        segment(a, 'kmeans', features=[0, 3])
    with pytest.raises(ValueError, match='out of range'):
        segment(a, 'otsu', reflectivity=5)


def test_segment_features_choose_the_clustered_contrasts():
    a = section()
    default = segment(a, 'kmeans', **PIXELS)
    assert np.array_equal(default, segment(a, 'kmeans', features=[0], **PIXELS))
    assert not np.array_equal(default, segment(a, 'kmeans', features=[1], **PIXELS))
    with pytest.raises(ValueError):
        segment(a, 'watershed')


def test_otsu_and_remove_small():
    values = np.concatenate([np.full(100, 0.2), np.full(100, 0.8)])
    assert 0.2 < otsu(values) < 0.8
    mask = np.zeros((10, 10), bool)
    mask[0, 0] = True
    mask[5:8, 5:8] = True
    kept = remove_small(mask, 2)
    assert not kept[0, 0] and kept[5:8, 5:8].all()


def test_segment_command_scores_and_records_settings(tmp_path):
    for i, name in enumerate(['ret', 'cro', 'ref']):
        np.save(tmp_path / f'{name}.npy', section()[..., i])
    Image.fromarray(expected_labels()).save(tmp_path / 'mask.png')
    with (tmp_path / 'manifest.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'section_id', 'split', 'image', 'mask'])
        w.writerow(['synthetic', '001', 'test', 'ret.npy;cro.npy;ref.npy', 'mask.png'])
    main(['segment', '--manifest', str(tmp_path / 'manifest.csv'), '--output', str(tmp_path / 'out'),
          '--method', 'kmeans', '--features', '0', '--pixel-size', '1', '--min-area', '20',
          '--min-fluid-area', '100'])  # fmt: skip
    saved = json.loads((tmp_path / 'out' / 'report.json').read_text())
    assert saved['method'] == 'kmeans' and saved['model_sha256'] is None
    assert saved['settings']['features'] == [0] and saved['settings']['min_fluid_area'] == 100
    assert saved['settings']['sample'] == 200_000 and saved['settings']['seed'] == 0  # defaults recorded too
    dice = saved['sections'][0]['metrics']['per_class']
    assert all(dice[c]['dice'] > 0.9 for c in ['gray_matter', 'white_matter', 'fluid_space'])
    report = run_classical(tmp_path / 'manifest.csv', 'test', tmp_path / 'out2', 'otsu', **PIXELS)
    assert report['settings']['features'] is None and report['mean_sample_macro_dice'] > 0.9
