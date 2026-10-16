# Brain Microscopy Tissue Segmentation

Develop and compare tissue segmentation methods for nonhuman primate brain microscopy, starting with polarization-sensitive optical coherence tomography (PS-OCT). This repository provides runnable **2D starter baselines, 3D Atropos pilots, and evaluation**, following the supplied project brief. LSFM, full-resolution Zarr, and deep-learning methods are planned extensions.

**Status:** Vlad is the first active sample. Three real MRI-aligned PS-OCT NIfTI contrasts and the lab's matching MRI brain mask have been downloaded and checked. Single-contrast and three-contrast 3D Atropos pilots have run. This follows the earlier Subject M JPEG pilot. Expert tissue annotations and anatomical segmentation accuracy remain unverified.

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

### Vlad 3D pilot

The active task follows the supplied `SegmentationProblem(3).docx`: begin with PS-OCT, adapt existing MRI segmentation methods, develop evaluation benchmarks, compare samples later, and extend to LSFM later. The first sample is **Vlad**, as requested.

```bash
pip install -r requirements-mri.txt
# Put the source files in data/vlad (outside Git):
# Ref_slide_deck_in_MRI.nii.gz
# Cro_slide_deck_in_MRI.nii.gz
# Ret_slide_deck_in_MRI.nii.gz
# nodif_brain_mask.nii.gz
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2 OPENBLAS_NUM_THREADS=1 python vlad_volume_pilot.py --mask data/vlad/nodif_brain_mask.nii.gz --output outputs/vlad_3d_mri_mask
```

Source: CMC - Sarah H / Derivatives / Vlad / PS-OCT / Enface / Registration. The brain mask is in Vlad / MRI / exvivo / proc / data. Native grid: 170 x 200 x 112, approximately 0.4 mm spacing. The default pilot downsamples by two to 85 x 100 x 56 (approximately 0.8 mm), with identical target geometry for all contrasts. The native affine, grid, finite values, and binary mask are checked before processing. The MRI mask is intersected with positive optical reflectivity support; this is a provisional support mask, not ground truth.

Two Gaussian-mixture Atropos runs use the same support and settings: reflectivity alone, then reflectivity + cross-polarization + retardance. Each contrast is clipped at its masked 1st/99th percentiles and scaled equally to 0.1..100.1. This numeric scaling avoids Atropos's absolute covariance-determinant floor on 0..1 multivariate inputs. No noise is added. The script passes seed, likelihood, and diagnostics consistently through the bundled Atropos entry point, because ANTsPy 0.6.3's multivariate wrapper drops optional kwargs. A constant seed is used.

Outputs include 3D class labels, posterior maps, support mask, source hashes, settings, quality flags, and views along three voxel axes. Posterior sums, finite probabilities, label range, and mask consistency are checked at runtime. **Adjusted Rand index compares the two methods' partitions; it is not accuracy.** A dominant-class flag requires review when one class covers more than 80% of the support.

The first MRI-mask pilot produced a dominant class of roughly 82% for both runs. This raises a concrete concern that intensity clustering is capturing broad signal groups rather than clean tissue types. Reference masks and review of native-resolution reflectivity/retardance are required before semantic class mapping or Dice/IoU benchmarking. Exact machine-readable diagnostics belong to each run's `report.json`.

### Source inventory checked October 16 2026

The lab project exposes seven named PS-OCT folders under Derivatives. Six contain data directories; **Legolas currently contains only a placeholder file**. MRI/PS-OCT coordinate agreement does not by itself establish registration accuracy.

| Folder | Observed PS-OCT inputs |
| --- | --- |
| Vlad | Numeric MAT sections, JPEGs, Zarr/OME-Zarr, registered NIfTI; first active sample |
| Moe | Numeric sections, JPEGs, Zarr, registration folders |
| KQRH | Numeric sections and Zarr |
| YoYo | Numeric cross-polarization, reflectivity, retardance sections |
| Zebel | Numeric sections and JPEG folders |
| Human_CX46009 | Human numeric tiles; separate species/domain, not pooled as another macaque |
| Legolas | Placeholder only; not counted as a usable dataset |

Directory listings are not a count of independent biological specimens. No LSFM count is established. Data access and contents may change.

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

**Confirmed target: four categories for every sample** — gray matter, white matter, fluid-filled spaces, and black space/background outside the tissue. The label IDs below stay consistent across samples. Uncertainty (255) is an annotation exclusion value, not a fifth tissue category.

| ID | Meaning |
| --- | --- |
| 0 | Black space / background outside tissue |
| 1 | Gray matter |
| 2 | White matter |
| 3 | Fluid-filled space |
| 255 | Unannotated or uncertain pixels, including excluded artifacts |

Use lossless PNG or NPY masks with the same shape and orientation as the image. Background is a valid annotated class; leave unannotated tissue at 255 rather than 0. Artifacts are excluded in this starter, not learned as a separate class. A domain expert must confirm labels and distinguish tissue, fluid spaces, tears, and other artifacts. MRI-derived labels require verified registration and expert review before use as references.

**Deferred review: dark tissue and orientation.** The project owner reports that tissue facing a particular direction can appear darker. Example review and confirmation of this effect are deferred. Darkness alone must not define background or fluid: ambiguous regions remain 255 in reference annotations until reviewed. Review the aligned reflectivity, cross-polarization, and retardance alongside spatial anatomy when this work resumes; these inputs do not yet establish a validated tissue mapping.

The existing three Atropos intensity groups plus zero outside the support mask are **not** a validated four-category anatomical segmentation. In particular, the current pilot intersects its MRI brain mask with positive reflectivity, so zero-valued or excluded dark tissue can fall outside its support. Do not interpret its zero label as confirmed background, or rename its three groups as gray matter, white matter, and fluid. No new tissue assignments or accuracy claims follow from this specification update.

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
