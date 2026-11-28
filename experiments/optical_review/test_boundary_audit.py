import numpy as np
from boundary_audit import audit


def test_faces_edges_and_unknown_centers():
    pred = np.zeros((5, 5, 5), np.uint8)
    ref = np.full_like(pred, 2)
    # A label on the opposite edge must not wrap into the neighborhood.
    pred[0, 2, 2] = pred[4, 2, 2] = pred[3, 2, 2] = pred[1, 1, 1] = 1
    ref[4, 2, 2] = 1
    ref[1, 1, 1] = 255
    codes, shifts = audit(pred, ref)
    assert codes[0, 2, 2] == 3
    assert codes[4, 2, 2] == 1
    assert codes[3, 2, 2] == 2
    assert codes[1, 1, 1] == 4
    assert len(shifts) == 6
    for row in shifts:
        assert row['scorable'] == 3
        assert row['shifted_matches'] == row['original_matches'] + row['gains'] - row['losses']
