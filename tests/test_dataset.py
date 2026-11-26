import json

import numpy as np
import pytest
from PIL import Image

from brainseg import IGNORE
from brainseg.benchmark import run_classical
from brainseg.cli import main
from brainseg.dataset import MRI, OPTICAL, build, section_names, stable, training_labels
from brainseg.io import load_row, read_manifest
from brainseg.mri_labels import consensus

nib = pytest.importorskip('nibabel')


def test_stable_needs_the_same_label_one_step_away_on_every_axis():
    a = np.zeros((7, 11), np.uint8)
    a[1:6, 2:9] = 1
    expected = np.zeros_like(a, bool)
    expected[2:5, 4:7] = True  # tissue at least (1, 2) pixels from any edge
    expected[[0, 0, 6, 6], [0, 10, 0, 10]] = expected[[0, 0, 6, 6], [1, 9, 1, 9]] = True  # background corners
    np.testing.assert_array_equal(stable(a, [1, 2]), expected)
    assert not stable(np.ones((3, 3), np.uint8), [5, 1]).any()  # outside the array counts as background


# One pixel per rule: (otsu, mri, stable, low signal, inside the optical outline) -> (label, source)
CASES = [
    ((1, 2, True, False, True), (1, OPTICAL)),  # otsu gray kept even where MRI disagrees
    ((2, IGNORE, False, False, True), (2, OPTICAL)),  # otsu white kept where MRI is uncertain
    ((1, 0, True, False, True), (1, OPTICAL)),  # otsu tissue beyond the MRI brain kept
    ((3, 2, True, False, True), (2, MRI)),  # otsu fluid (often gel gaps) not trusted: stable MRI used
    ((0, 1, True, False, True), (1, MRI)),  # dark gap inside the tissue: stable MRI fill
    ((0, 1, True, False, False), (IGNORE, IGNORE)),  # gel around the brain is never filled, even stable MRI
    ((0, 1, False, False, True), (IGNORE, IGNORE)),  # unstable MRI fill is not learned from
    ((0, IGNORE, True, False, True), (IGNORE, IGNORE)),  # uncertain MRI is not learned from
    ((0, 3, True, False, True), (3, MRI)),  # fluid comes from MRI
    ((0, 3, False, False, True), (3, MRI)),  # MRI fluid in an enclosed optical gap counts even when not stable
    ((3, 3, False, False, True), (3, MRI)),  # likewise where otsu calls the gap fluid
    ((0, 3, False, False, False), (IGNORE, IGNORE)),  # but not MRI fluid on the gel around the brain
    ((0, 3, False, True, True), (IGNORE, IGNORE)),  # nor where the dark pixel is signal dropout
    ((1, 3, False, False, True), (1, OPTICAL)),  # otsu tissue is kept over unstable MRI fluid
    ((0, 0, True, False, False), (0, 0)),  # background where both agree
    ((1, 2, True, True, True), (2, MRI)),  # dropout: optical not trusted, stable MRI used
    ((0, 1, True, True, False), (1, MRI)),  # dropout can break the optical outline: still filled
    ((0, 0, True, True, False), (IGNORE, IGNORE)),  # dropout outside the brain: not background
]


def test_assisted_labels_follow_each_rule():
    otsu, mri, stab, low, inside = (np.array([[c[0][k] for c in CASES]]) for k in range(5))
    labels, source = training_labels(otsu.astype(np.uint8), mri.astype(np.uint8), stab, low, inside)
    assert labels[0].tolist() == [c[1][0] for c in CASES]
    assert source[0].tolist() == [c[1][1] for c in CASES]


def test_consensus_mode_and_unknown_mode():
    otsu = np.array([[1, 2, 0, 1, 3]], np.uint8)
    mri = np.array([[1, 1, 0, 0, 3]], np.uint8)
    ones, zeros = np.ones_like(otsu, bool), np.zeros_like(otsu, bool)
    labels, source = training_labels(otsu, mri, ones, zeros, ones, 'consensus')
    np.testing.assert_array_equal(labels, consensus(mri, otsu))
    assert source[0].tolist() == [OPTICAL, IGNORE, 0, IGNORE, OPTICAL]
    with pytest.raises(ValueError, match='unknown mode'):
        training_labels(otsu, mri, ones, zeros, ones, 'majority')


def test_section_names(tmp_path):
    mapping = tmp_path / 'map.json'
    mapping.write_text(json.dumps({'0': 'Slice_327_*_hdr.nii.gz', '1': 'Slice_326_*_hdr.nii.gz'}))
    assert section_names(mapping, 3) == ['Slice_327', 'Slice_326', '002']
    assert section_names(None, 2) == ['000', '001']


def deck(tmp_path, floor_region=True):
    """Coronal-like deck (axis 1 = 8 sections) at 0.055 x 0.2 x 0.055 mm: a tissue disk in gel, and its MRI labels.

    Reflectivity is in dB as in the real decks (gel 40, tissue 60); the dropout floor 32.5 is darker than gel.
    """
    yy, xx = np.mgrid[:80, :80]
    r = np.hypot(yy - 40, xx - 40)
    tissue, white = r < 34, r < 16
    ref = np.where(tissue, 60.0, 40.0).astype(np.float32)
    ret = np.where(white, 0.9, np.where(tissue, 0.3, 0.1)).astype(np.float32)
    if floor_region:
        quadrant = np.zeros_like(tissue)
        quadrant[40:, 40:] = True
        ref[quadrant & tissue] = 32.5  # dropout inside the tissue
        ref[74:78, 2:8] = 32.5  # and in the gel outside the brain
    mri = np.select([white, tissue], [2, 1], 0).astype(np.uint8)
    ref[tissue & (xx < 20)] = 40.0  # optical tissue edge 14 px inside the MRI outline: gel there to the PS-OCT
    ret, ref, mri = (np.repeat(a[:, None], 8, 1) for a in (ret, ref, mri))
    ret[:3], ref[:3] = 0, 0  # unimaged border
    mri[:, 0] = 0  # first section has no MRI tissue
    affine = np.diag([0.055, 0.2, 0.055, 1])
    paths = {}
    for name, a in (('ret', ret), ('ref', ref), ('mri', mri)):
        paths[name] = tmp_path / f'{name}.nii.gz'
        nib.save(nib.Nifti1Image(a, affine), paths[name])
    mapping = tmp_path / 'map.json'
    mapping.write_text(json.dumps({str(i): f'Slice_{100 + i:03d}_*_hdr.nii.gz' for i in range(8)}))
    return paths, mapping


def test_build_writes_a_manifest_ready_training_set(tmp_path):
    paths, mapping = deck(tmp_path)
    out = tmp_path / 'set'
    report = build(paths['ret'], paths['ref'], paths['mri'], out, 'moe', 1, floor=32.5, mapping=mapping)
    assert report['sections'] == 7 and report['stability_steps_px'] == [7, 2, 7]
    rows = read_manifest(out / 'manifest.csv')
    assert [r['section_id'] for r in rows] == [f'Slice_{100 + i}' for i in range(1, 8)]
    assert rows[0]['pixel_size_mm'] == '0.055' and load_row(rows[0]).shape == (80, 80, 2)
    labels = np.asarray(Image.open(out / 'labels' / 'moe_Slice_104.png'))
    source = np.asarray(Image.open(out / 'sources' / 'moe_Slice_104.png'))
    assert (labels[:3] == IGNORE).all() and (source[:3] == IGNORE).all()  # unimaged
    assert labels[34, 40] == 2 and source[34, 40] == OPTICAL  # white core outside the dropout, optical
    assert labels[58, 58] == 1 and source[58, 58] == MRI  # dropout, stable MRI gray matter
    assert labels[62, 63] == IGNORE  # dropout next to the tissue edge: MRI not stable, optical not trusted
    assert labels[10, 75] == 0 and source[10, 75] == 0  # gel where both agree
    assert labels[40, 14] == IGNORE  # gel the stable MRI outline passes: never filled
    assert labels[76, 5] == IGNORE  # dropout outside the brain is not taught as background
    first = np.asarray(Image.open(out / 'labels' / 'moe_Slice_101.png'))
    assert first[58, 58] == IGNORE  # through-plane neighbour two sections away has no MRI tissue: not stable
    middle = report['per_section'][3]
    assert sum(middle[k] for k in ('background', 'optical', 'mri', 'ignored')) == 80 * 80
    assert report['totals']['mri'] == sum(s['mri'] for s in report['per_section']) > 0
    with pytest.raises(FileExistsError):
        build(paths['ret'], paths['ref'], paths['mri'], out, 'moe', 1)


def test_build_output_scores_with_the_benchmark(tmp_path):
    paths, mapping = deck(tmp_path, floor_region=False)
    build(paths['ret'], paths['ref'], paths['mri'], tmp_path / 'set', 'moe', 1, split='test', mode='consensus')
    report = run_classical(tmp_path / 'set' / 'manifest.csv', 'test', tmp_path / 'scored', 'otsu', reflectivity=1)
    assert len(report['sections']) == 7 and report['samples']['moe']['hd95_um'] is not None


def test_build_dataset_command(tmp_path, capsys):
    paths, mapping = deck(tmp_path)
    main(['build-dataset', '--retardance', str(paths['ret']), '--reflectivity', str(paths['ref']),
          '--mri-labels', str(paths['mri']), '--sample-id', 'moe', '--axis', '1', '--output', str(tmp_path / 's'),
          '--floor', '32.5', '--mapping', str(mapping), '--split', 'val'])  # fmt: skip
    printed = json.loads(capsys.readouterr().out)
    assert printed['split'] == 'val' and printed['floor'] == 32.5 and 'per_section' not in printed
    assert {r['split'] for r in read_manifest(tmp_path / 's' / 'manifest.csv')} == {'val'}
