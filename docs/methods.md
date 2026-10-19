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
