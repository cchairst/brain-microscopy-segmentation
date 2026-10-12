import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat
from segmentation import metrics, read_array, read_manifest, read_mask, normalize, demo, run_predictions


class SegmentationTests(unittest.TestCase):
    def test_exact_metrics_and_absent_classes(self):
        cm = np.array([[5, 0, 0, 0], [0, 3, 1, 0], [0, 1, 3, 0], [0, 0, 0, 0]])
        m = metrics(cm)
        self.assertAlmostEqual(m['per_class']['gray_matter']['dice'], .75)
        self.assertAlmostEqual(m['per_class']['white_matter']['iou'], .6)
        self.assertIsNone(m['per_class']['fluid_space']['dice'])
        self.assertAlmostEqual(m['macro_tissue']['dice'], .75)

    def test_background_false_positives_count(self):
        cm = np.zeros((4, 4), int); cm[0, 1] = 2; cm[1, 1] = 2
        self.assertAlmostEqual(metrics(cm)['per_class']['gray_matter']['precision'], .5)

    def test_io_and_mask_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'section.mat'; a = np.arange(12).reshape(3, 4)
            savemat(p, {'intensity': a})
            np.testing.assert_array_equal(read_array(p, 'intensity'), a)
            with self.assertRaises(ValueError): read_array(p)
            p = Path(d) / 'mask.npy'; np.save(p, np.full((3, 4), 255))
            self.assertTrue((read_mask(p, (3, 4)) == 255).all())
            np.save(p, np.full((3, 4), 8))
            with self.assertRaises(ValueError): read_mask(p, (3, 4))
            np.testing.assert_array_equal(normalize(np.ones((3, 4))), 0)

    def test_end_to_end_and_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            demo(d)
            manifest = Path(d) / 'manifest.csv'
            rows = read_manifest(manifest)
            for method in ['kmeans', 'rf']:
                report = run_predictions(manifest, Path(d) / f'{method}.joblib', 'test', Path(d) / 'check' / method)
                self.assertEqual(len(report['samples']), 2)
                self.assertGreater(report['mean_sample_macro_dice'], .9)
            # Separate manifest still cannot evaluate a trained sample as a test sample.
            with manifest.open(newline='') as f: original = list(csv.DictReader(f))
            changed = [dict(original[0], split='test')]
            other = Path(d) / 'leaked.csv'
            with other.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(changed[0])); w.writeheader(); w.writerows(changed)
            with self.assertRaises(ValueError): run_predictions(other, Path(d) / 'rf.joblib', 'test', Path(d) / 'bad')
            original[-1]['sample_id'] = original[0]['sample_id']
            with manifest.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(original[0])); w.writeheader(); w.writerows(original)
            with self.assertRaises(ValueError): read_manifest(manifest)


if __name__ == '__main__':
    unittest.main()
