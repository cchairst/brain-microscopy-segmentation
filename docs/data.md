# Data and labels

## Source inventory checked October 16 2026

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
brainseg train --manifest manifest.csv --method kmeans --output outputs/kmeans.joblib
brainseg train --manifest manifest.csv --method rf --output outputs/rf.joblib
brainseg predict --manifest manifest.csv --model outputs/rf.joblib --split val --output outputs/rf_val
brainseg predict --manifest manifest.csv --model outputs/rf.joblib --split test --output outputs/rf_test
```

Image inputs: numeric 2D NPY, MATLAB v4-v7.2 with an explicit `mat_key`, or 2D image files. For H x W x C images, put a zero-based channel index in the `channel` column. To use several contrasts as features, list one image per contrast in the `image` column, separated by `;` (for example `ret.npy;cro.npy;ref.npy`); `mat_key` and `channel` take either one value for all images or one per image in the same order. All contrasts in a row must share a shape, and every row must use the same number of contrasts. MATLAB v7.3/HDF5, whole 3D volumes, and Zarr are not implemented; convert or export verified 2D sections first. This code loads each section into memory, so start with downsampled sections.

Both initial methods require train masks; K-means itself clusters intensities without labels, but semantic cluster mapping needs training references. Prediction rows may leave `mask` empty. They generate predictions without scores. `joblib` model files can execute code: load only models you created or trust.
