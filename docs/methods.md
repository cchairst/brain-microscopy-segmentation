# Methods

## Vlad 3D pilot

The active task follows the supplied `SegmentationProblem(3).docx`: begin with PS-OCT, adapt existing MRI segmentation methods, develop evaluation benchmarks, compare samples later, and extend to LSFM later. The first sample is **Vlad**, as requested.

```bash
pip install -e ".[mri]"
# Put the source files in data/vlad (outside Git):
# Ref_slide_deck_in_MRI.nii.gz
# Cro_slide_deck_in_MRI.nii.gz
# Ret_slide_deck_in_MRI.nii.gz
# nodif_brain_mask.nii.gz
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2 OPENBLAS_NUM_THREADS=1 brainseg vlad-pilot --mask data/vlad/nodif_brain_mask.nii.gz --output outputs/vlad_3d_mri_mask
```

Source: CMC - Sarah H / Derivatives / Vlad / PS-OCT / Enface / Registration. The brain mask is in Vlad / MRI / exvivo / proc / data. Native grid: 170 x 200 x 112, approximately 0.4 mm spacing. The default pilot downsamples by two to 85 x 100 x 56 (approximately 0.8 mm), with identical target geometry for all contrasts. The native affine, grid, finite values, and binary mask are checked before processing. The MRI mask is intersected with positive optical reflectivity support; this is a provisional support mask, not ground truth.

Two Gaussian-mixture Atropos runs use the same support and settings: reflectivity alone, then reflectivity + cross-polarization + retardance. Each contrast is clipped at its masked 1st/99th percentiles and scaled equally to 0.1..100.1. This numeric scaling avoids Atropos's absolute covariance-determinant floor on 0..1 multivariate inputs. No noise is added. The script passes seed, likelihood, and diagnostics consistently through the bundled Atropos entry point, because ANTsPy 0.6.3's multivariate wrapper drops optional kwargs. A constant seed is used.

Outputs include 3D class labels, posterior maps, support mask, source hashes, settings, quality flags, and views along three voxel axes. Posterior sums, finite probabilities, label range, and mask consistency are checked at runtime. **Adjusted Rand index compares the two methods' partitions; it is not accuracy.** A dominant-class flag requires review when one class covers more than 80% of the support.

The first MRI-mask pilot produced a dominant class of roughly 82% for both runs. This raises a concrete concern that intensity clustering is capturing broad signal groups rather than clean tissue types. Reference masks and review of native-resolution reflectivity/retardance are required before semantic class mapping or Dice/IoU benchmarking. Exact machine-readable diagnostics belong to each run's `report.json`.

## MRI segmentation tool baseline

`brainseg atropos` (`brainseg.methods.atropos`) applies **ANTs Atropos**, a medical-image finite-mixture segmentation method with Markov random field smoothing, to PS-OCT scalar images. It supports 2D grayscale JPEG/PNG pilots and numeric 2D/3D NIfTI with a reviewed binary foreground mask. It uses no human atlas or pretrained human MRI network.

```bash
pip install -e ".[mri]"
brainseg atropos --image path/to/Slice_110_Ref.jpeg --output outputs/atropos_pilot
# For a numeric NIfTI volume, provide a reviewed mask on the same grid:
brainseg atropos --image path/to/Ref_slide_deck.nii.gz --mask path/to/foreground_mask.nii.gz --output outputs/atropos_volume
```

Outputs include raw class labels, posterior probability volumes, a foreground mask, source hash, settings, class counts, and a 2D preview. **Atropos class IDs represent intensity groups; they are not the semantic tissue IDs used by the benchmark (`brainseg train`/`predict`).** Do not pass these labels into the tissue benchmark until an expert establishes a mapping using training references. Fluid spaces and tears may have similar intensity.

The JPEG pilot is downsampled to a maximum side of 1100 pixels. Its default foreground mask uses a threshold of 20/255, closing, hole filling, and the largest connected component; it is unreviewed and can omit disconnected tissue or include gaps. Supply a reviewed mask for research runs. NIfTI retains physical metadata and requires an explicit mask. No MRI bias-field correction is imposed on optical contrasts.

Real-data pilot: Subject M (Moe), reflectivity section 110, from [the CMC PS-OCT publication](https://brainlife.io/pub/CMC#tab-ps-oct). JPEG input is 11000 x 8500 pixels. Public images were accessible, but the source project required membership. Expert tissue reference masks have not been located, so no real-data Dice/IoU results are reported. This pilot tests transfer of an MRI segmentation algorithm to optical images; its tissue accuracy is unvalidated.

| Method | Input features | Tissue interpretation |
| --- | --- | --- |
| `kmeans` | Normalized scalar intensity | Four intensity clusters mapped by majority class using training annotations only; clusters can map to the same class |
| `rf` | Intensity, Gaussian scales 1 and 3, 5-pixel local mean and standard deviation, smoothed gradient magnitude | Supervised random forest with class balancing |

Intensity is clipped and scaled using each section's 1st/99th percentiles. This is a fixed per-image preprocessing rule; assess whether it removes meaningful across-sample contrast before scientific use. Feature scales are in pixels and require calibration to voxel size. RGB JPEGs may be colorized visualizations: select a scientifically meaningful scalar channel explicitly; do not treat arbitrary color channels as quantitative PS-OCT measurements.

## Classical (unsupervised) segmentation

`brainseg segment` labels each section of a manifest split without training labels and scores it like `brainseg predict` wherever a reference mask exists. List the contrasts in each row's `image` field and point `--retardance` and `--reflectivity` at them (defaults: first and last, e.g. `ret.npy;cro.npy;ref.npy`).

```bash
brainseg segment --manifest manifest.csv --split test --method gmm --output outputs/gmm_test
```

1. **Tissue:** Otsu threshold on reflectivity, computed over imaged pixels only (NaN and zero-valued pixels outside the scan are excluded so they cannot drag the threshold into the gel). Tissue islands smaller than `--min-size` pixels are dropped.
2. **Fluid:** holes enclosed by tissue of at least `--min-fluid` pixels. Smaller holes, such as vessels, are filled with the surrounding tissue label, matching the labeling rules.
3. **Gray/white split** inside the tissue, after rescaling each contrast to the tissue's own 1st/99th percentiles: `otsu` thresholds retardance; `gmm` and `kmeans` fit two clusters on the `--features` contrasts (default retardance only) using at most 200,000 sampled pixels. White matter is the higher-retardance group.
4. **Cleanup:** white or gray islands smaller than `--min-size` pixels are merged into the other class.

Report `settings` record the method and every parameter. Damage (4) is not predicted.

**Feature choice (exploration on 10 low-res sections per subject, no labels yet).** On retardance alone all three variants give stable white-matter fractions (Otsu and k-means 15-47% of tissue; GMM up to about 15 points higher because it allows unequal class spreads). Adding cross-polarization, or reflectivity, makes clustering split along variation unrelated to gray/white on Moe (91-98% "white"), so the default clusters on retardance only. `--features 0,1` reproduces the comparison; it is a channel ablation to repeat once reference labels exist.

**Known limitations.** Enclosed gel gaps (for example between cerebellar folia) are labeled fluid when larger than `--min-fluid`. `--min-size` and `--min-fluid` are in pixels and default to values for the 10x downsampled sections (55 um); scale them by about 100 for full resolution. GMM is unstable on nearly empty end sections.

## Benchmark outputs

- Per-class Dice, IoU, precision, recall, support, and a 5 x 5 confusion matrix in ID order 0-4.
- Per-class HD95 (95th-percentile symmetric surface distance) in pixels for each section; per-brain HD95 is the mean of its sections. Tissue volume fractions (classes 1-3) for reference and prediction.
- Tissue macro metrics and volume fractions cover gray matter, white matter and fluid; they exclude background and damage but include mistakes to/from them.
- Pixels marked 255 in the reference are ignored. Undefined ratios are JSON `null`; macros average defined classes. Always inspect class support before comparing scores.
- Per-brain metrics pool that brain's annotated sections; mean sample macro Dice weights each annotated brain equally. Pooled scores also appear and can favor larger samples.
- Model metadata stores train sample IDs, train image/mask hashes, manifest hash, seed, sampling budget, and dependency versions. Reports store image/model/manifest hashes. Preserve both the model and the original manifest with each run.

## Next research phases

1. Establish expert reference masks and inter-rater agreement on representative sections across brains.
2. Run these baselines and inspect failures at tissue boundaries and artifacts.
3. Add supervised U-Net/nnU-Net after confirming training volume, label consistency, and compute availability; compare against the same frozen split.
4. Extend to aligned 3D PS-OCT with physical spacing and boundary metrics.
5. Add chunked Zarr processing and LSFM with explicit distortion/registration checks; compare samples and modalities separately.
