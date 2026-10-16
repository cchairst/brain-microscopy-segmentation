# Vlad PS OCT 3D pilot

This is a feasibility pilot for the supplied tissue segmentation project, with Vlad as the first sample. It compares two input configurations of ANTs Atropos. Anatomical tissue labels and accuracy are unvalidated.

## Data and processing

The source images are MRI aligned reflectivity, cross polarization, and retardance NIfTI volumes from the lab project. All three inputs have the same native grid and affine and contain finite numeric values. The existing MRI binary brain mask also matches. The native 170 x 200 x 112 grid is downsampled to 85 x 100 x 56 for this pilot, at approximately 0.8 mm spacing. The mask is intersected with positive reflectivity support.

## Observed results

| Input configuration | Class 1 | Class 2 | Class 3 |
| --- | ---: | ---: | ---: |
| reflectivity only | 3.75% | 14.27% | 81.98% |
| three contrasts | 3.25% | 15.17% | 81.57% |

Fractions describe the segmentation support, not measured tissue composition. Class IDs are intensity groups with no verified mapping to gray matter, white matter, or fluid spaces.

The adjusted Rand index between the two partitions is 0.8225. This measures agreement between methods, not accuracy. Both configurations have a dominant class above 80 percent, so the output is flagged for review.

## Validation and limits

Runtime checks passed for source geometry, finite values, binary mask, output label range, mask consistency, and posterior sums. The existing four baseline tests also pass. Reference tissue masks have not been located; Dice and IoU remain unavailable. A matching coordinate grid does not prove the underlying registration is accurate. The brain mask is not anatomical tissue ground truth.

The report JSON records input hashes, dependency versions, preprocessing, method settings, and diagnostic statistics. The preview shows the inputs and output classes along three voxel axes. Native resolution review and expert tissue annotations are required before scientific performance conclusions.

## Next work

The target for Vlad and every subsequent sample is four semantic categories: 0 black space/background outside tissue, 1 gray matter, 2 white matter, and 3 fluid-filled spaces. Value 255 is reserved for unannotated or uncertain reference regions and excluded artifacts. These target IDs do not map directly to the current Atropos output IDs.

The project owner noted dark tissue associated with its direction/orientation. Inspection of examples is deferred. Until reviewed, do not use darkness alone to assign fluid or background; uncertain reference regions stay at 255. The current positive-reflectivity support can exclude dark or zero-valued tissue, and output zero means outside the pilot support, not verified anatomical background. Revisit support and tissue mapping during the deferred review using aligned contrasts and anatomical context.

Review Vlad's native optical sections alongside these MRI aligned outputs, check registration and mask boundaries, and create expert reference annotations. Then compare multiple segmentation methods against the same annotations. Extend to other samples after the Vlad workflow is validated. LSFM remains a later phase.
