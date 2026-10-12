# Brain Microscopy Tissue Segmentation

Develop and compare tissue segmentation methods for nonhuman primate brain microscopy, starting with polarization-sensitive optical coherence tomography (PS-OCT). This repository provides runnable **2D starter baselines and evaluation**, following the supplied project brief. LSFM, aligned 3D models, full-resolution Zarr, and deep-learning methods are planned extensions.

**Status:** K-means and random forest are tested on synthetic data. ANTs Atropos has also run on a real, downsampled Subject M reflectivity section from the CMC publication. Expert annotations, quantitative channel scaling, and cross-modality alignment remain unverified. No real-data tissue accuracy claims are made.

## Quick start

Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest -v test_segmentation.py
python segmentation.py demo --output outputs/synthetic_demo
```

The demo creates six independent synthetic brain-like sections, trains both methods on three samples, and evaluates two held-out samples. The fourth sample is reserved for validation. Output masks use integer class IDs; `report.json` contains section, sample, and pooled scores. High scores on this easy synthetic example are software checks, not evidence of microscopy performance.

## Methods

### MRI segmentation tool baseline

`mri_baseline.py` applies **ANTs Atropos**, a medical-image finite-mixture segmentation method with Markov random field smoothing, to PS-OCT scalar images. It supports 2D grayscale JPEG/PNG pilots and numeric 2D/3D NIfTI with a reviewed binary foreground mask. It uses no human atlas or pretrained human MRI network.

```bash
pip install -r requirements-mri.txt
python mri_baseline.py --image path/to/Slice_110_Ref.jpeg --output outputs/atropos_pilot
# For a numeric NIfTI volume, provide a reviewed mask on the same grid:
python mri_baseline.py --image path/to/Ref_slide_deck.nii.gz --mask path/to/foreground_mask.nii.gz --output outputs/atropos_volume
```

Outputs include raw class labels, posterior probability volumes, a foreground mask, source hash, settings, class counts, and a 2D preview. **Atropos class IDs represent intensity groups; they are not the semantic tissue IDs used by `segmentation.py`.** Do not pass these labels into the tissue benchmark until an expert establishes a mapping using training references. Fluid spaces and tears may have similar intensity.

The JPEG pilot is downsampled to a maximum side of 1100 pixels. Its default foreground mask uses a threshold of 20/255, closing, hole filling, and the largest connected component; it is unreviewed and can omit disconnected tissue or include gaps. Supply a reviewed mask for research runs. NIfTI retains physical metadata and requires an explicit mask. No MRI bias-field correction is imposed on optical contrasts.

Real-data pilot: Subject M (Moe), reflectivity section 110, from [the CMC PS-OCT publication](https://brainlife.io/pub/CMC#tab-ps-oct). JPEG input is 11000 x 8500 pixels. Public images were accessible, but the source project required membership. Expert tissue reference masks have not been located, so no real-data Dice/IoU results are reported. This pilot tests transfer of an MRI segmentation algorithm to optical images; its tissue accuracy is unvalidated.

| Method | Input features | Tissue interpretation |
| --- | --- | --- |
| `kmeans` | Normalized scalar intensity | Four intensity clusters mapped by majority class using training annotations only; clusters can map to the same class |
| `rf` | Intensity, Gaussian scales 1 and 3, 5-pixel local mean and standard deviation, smoothed gradient magnitude | Supervised random forest with class balancing |

Intensity is clipped and scaled using each section's 1st/99th percentiles. This is a fixed per-image preprocessing rule; assess whether it removes meaningful across-sample contrast before scientific use. Feature scales are in pixels and require calibration to voxel size. RGB JPEGs may be colorized visualizations: select a scientifically meaningful scalar channel explicitly; do not treat arbitrary color channels as quantitative PS-OCT measurements.

## Labels and annotations

| ID | Meaning |
| --- | --- |
| 0 | Background outside tissue |
| 1 | Gray matter |
| 2 | White matter |
| 3 | Fluid-filled space |
| 255 | Unannotated or uncertain pixels, including excluded artifacts |

Use lossless PNG or NPY masks with the same shape and orientation as the image. Background is a valid annotated class; leave unannotated tissue at 255 rather than 0. Artifacts are excluded in this starter, not learned as a separate class. A domain expert must confirm labels and distinguish tissue, fluid spaces, tears, and other artifacts. MRI-derived labels require verified registration and expert review before use as references.

## Use real data

1. Confirm the original Brainlife project, sample IDs, dataset IDs, permissions, channel definitions, and pixel spacing with the data owner. Preserve the read-only source. Keep images and annotations outside Git.
2. Copy `manifest.example.csv` to `manifest.csv`; update its paths and IDs. Paths are relative to the manifest directory. The example is a schema, not a downloadable dataset.
3. Assign **whole brains** to `train`, `val`, or `test`. Sections, blocks, repeat scans, and aligned modalities from one brain must share its sample ID and split. The code rejects sample overlap and duplicate image paths but cannot detect misidentified brains or duplicated content under different paths. If only one brain is available, use a clearly labeled spatial pilot and do not claim cross-brain generalization.
4. Train using labeled train rows only, tune choices on validation samples, then run the untouched test set once.

```bash
python segmentation.py train --manifest manifest.csv --method kmeans --output outputs/kmeans.joblib
python segmentation.py train --manifest manifest.csv --method rf --output outputs/rf.joblib
python segmentation.py predict --manifest manifest.csv --model outputs/rf.joblib --split val --output outputs/rf_val
python segmentation.py predict --manifest manifest.csv --model outputs/rf.joblib --split test --output outputs/rf_test
```

Image inputs: numeric 2D NPY, MATLAB v4-v7.2 with an explicit `mat_key`, or 2D image files. For H x W x C images, put a zero-based channel index in the `channel` column. MATLAB v7.3/HDF5, whole 3D volumes, and Zarr are not implemented; convert or export verified 2D sections first. This code loads each section into memory, so start with downsampled sections.

Both initial methods require train masks; K-means itself clusters intensities without labels, but semantic cluster mapping needs training references. Prediction rows may leave `mask` empty. They generate predictions without scores. `joblib` model files can execute code: load only models you created or trust.

## Benchmark outputs

- Per-class Dice, IoU, precision, recall, support, and a 4 x 4 confusion matrix in ID order 0,1,2,3.
- Tissue macro metrics exclude background but include mistakes to/from background.
- Pixels marked 255 in the reference are ignored. Undefined ratios are JSON `null`; macros average defined classes. Always inspect class support before comparing scores.
- Per-brain metrics pool that brain's annotated sections; mean sample macro Dice weights each annotated brain equally. Pooled scores also appear and can favor larger samples.
- Model metadata stores train sample IDs, train image/mask hashes, manifest hash, seed, sampling budget, and dependency versions. Reports store image/model/manifest hashes. Preserve both the model and the original manifest with each run.

## Next research phases

1. Establish expert reference masks and inter-rater agreement on representative sections across brains.
2. Run these baselines and inspect failures at tissue boundaries and artifacts.
3. Add supervised U-Net/nnU-Net after confirming training volume, label consistency, and compute availability; compare against the same frozen split.
4. Extend to aligned 3D PS-OCT with physical spacing and boundary metrics.
5. Add chunked Zarr processing and LSFM with explicit distortion/registration checks; compare samples and modalities separately.

## Documentation

- [Brainlife project access and data organization](https://brainlife.io/docs/user/project/)
- [SciPy MATLAB reader](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html)
- [MiniBatchKMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)
- [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

No research images, private source brief, or access credentials are included. The repository is initially private and has no open-source license selected.
