# Brain Microscopy Tissue Segmentation

Develop and compare tissue segmentation methods for nonhuman primate brain microscopy, starting with polarization-sensitive optical coherence tomography (PS-OCT). This repository provides runnable **2D starter baselines, 3D Atropos pilots, and evaluation**, following the supplied project brief. LSFM, full-resolution Zarr, and deep-learning methods are planned extensions.

**Status:** Vlad is the first active sample. Three real MRI-aligned PS-OCT NIfTI contrasts and the lab's matching MRI brain mask have been downloaded and checked. Single-contrast and three-contrast 3D Atropos pilots have run. This follows the earlier Subject M JPEG pilot. Expert tissue annotations and anatomical segmentation accuracy remain unverified.

## Quick start

Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"        # add ,mri for the ANTs Atropos commands
pytest
brainseg demo --output outputs/synthetic_demo
```

The demo creates six independent synthetic brain-like sections, trains both methods on three samples, and evaluates two held-out samples. The fourth sample is reserved for validation. Output masks use integer class IDs; `report.json` contains section, sample, and pooled scores. High scores on this easy synthetic example are software checks, not evidence of microscopy performance.

## Commands

| Command | What it does |
| --- | --- |
| `brainseg train` | Train the k-means or random forest baseline on a manifest's train rows |
| `brainseg predict` | Predict a manifest split and score it against reference masks |
| `brainseg demo` | Synthetic end-to-end software check |
| `brainseg atropos` | ANTs Atropos intensity classes on a 2D section or 3D NIfTI (needs `.[mri]`) |
| `brainseg vlad-pilot` | Single- vs three-contrast 3D Atropos on MRI-aligned Vlad volumes (needs `.[mri]`) |

Run `brainseg <command> --help` for options.

## Layout

```
src/brainseg/
  io.py            read images, masks and manifests; file hashes
  preprocess.py    percentile clipping and scaling
  metrics.py       Dice, IoU, precision, recall from a confusion matrix
  benchmark.py     predict + score a manifest split; synthetic demo
  methods/
    baselines.py   k-means and random forest pixel classifiers
    atropos.py     ANTs Atropos on one section or volume
  vlad_pilot.py    Vlad 3D Atropos pilot
  cli.py           brainseg command
tests/
docs/
  methods.md       methods, benchmark outputs, research phases
  data.md          source inventory, labels and annotations, using real data
  vlad_pilot.md    Vlad pilot results
```

## Labels

| ID | Meaning |
| --- | --- |
| 0 | Black space / background outside tissue |
| 1 | Gray matter |
| 2 | White matter |
| 3 | Fluid-filled space |
| 255 | Unannotated or uncertain pixels, including excluded artifacts |

Details, annotation rules and the deferred dark-tissue review are in [docs/data.md](docs/data.md).

## Documentation

- [Methods](docs/methods.md), [data and labels](docs/data.md), [Vlad pilot](docs/vlad_pilot.md)
- [Brainlife project access and data organization](https://brainlife.io/docs/user/project/)
- [SciPy MATLAB reader](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html)
- [MiniBatchKMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)
- [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

No research images, private source brief, or access credentials are included. No open-source license is selected.
