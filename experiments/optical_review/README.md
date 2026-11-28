# Optical review of unresolved tissue

This experiment compares a relaxed optical threshold and a random forest on near-threshold voxels. Each forest trains on one animal and predicts the other. The paired comparison evaluates the two methods on the same previously unresolved voxels with an available MRI reference class. Agreement measures consistency with that MRI segmentation. Anatomical accuracy requires independent tissue annotations.

## Run on local data

From the repository root, install the package with `python -m pip install -e ".[mri]"`. Prepare the registered reflectivity and retardance volumes and the notebook 02 semantic MRI reference for each animal, named `MoeRef.nii.gz`, `MoeRet.nii.gz`, `Moe_MRI_reference.nii.gz` and their `Vlad` counterparts. This trial expects the documented padded optical registration for Moe and the MRI-sized registration for Vlad.

Set `inputs` to the directory holding those six volumes and `current` to the parent of the `Moe` and `Vlad` output folders from `brainseg assist-volume`. The experiment reads each animal’s tissue mask, source map and priority slice from that run. See [the whole-brain workflow](../../docs/assisted_sections.md#whole-brain-output-on-the-mri-grid).

```bash
inputs=/path/to/registered_volumes
current=/path/to/whole_brain_outputs
out=outputs/optical_review
trial=experiments/optical_review

python "$trial/threshold_trial.py" --inputs "$inputs" --current "$current" --output "$out"
python "$trial/borderline_forest.py" --inputs "$inputs" --current "$current" --output "$out"
python "$trial/paired_comparison.py" --inputs "$inputs" --current "$current" --output "$out"
python "$trial/review_output.py" --inputs "$inputs" --current "$current" --output "$out"
```

## Inspect a run

Open the generated `Moe_review.html` or `Vlad_review.html` in a browser. Each viewer pairs optical reflectivity with the current tissue mask or the binary review flags. A flag marks an unresolved voxel where the nearest optical tissue labels on both sides support its proposed class along at least two axes. The separate experimental forest candidate maps use 1 for gray and 2 for white. A forest score threshold of 0.95 selects proposals. That score has no independent probability calibration.

| Local output | Contents |
| --- | --- |
| `results.csv` | Threshold candidates before and after requiring spatial support, including proposals outside the MRI outline |
| `forest_results.csv` | Cross-animal forest coverage and agreement among eligible unresolved voxels |
| `paired_results.csv` | Both methods scored on identical held-out voxels, including cases where only one agrees with MRI |
| `*_review_flags.nii.gz` | Binary spatial review flags on the MRI grid |
| `*_experimental_candidates.nii.gz` | Forest class proposals on previously unresolved voxels |
| `BCM_optical_review.png` | Side-by-side preview using each animal’s slice with the most review flags |

## Check sensitivity to MRI alignment

Run this after the forest comparison, once per animal.

```bash
python "$trial/boundary_audit.py" --candidates "$out/Vlad_experimental_candidates.nii.gz" \
  --mri "$inputs/Vlad_MRI_reference.nii.gz" --output "$out/Vlad_boundary"
```

The audit map uses 0 for no proposal, 1 for an exact match, 2 for a mismatch whose proposed class appears in at least one of six face neighbours, 3 for a mismatch with no such neighbour, and 4 for an unscorable MRI center. For a uniform shift, `shift_scores.csv` reports gains and losses on the same scorable centers at every offset. Axis and step are MRI voxel coordinates. Out-of-volume samples count as background. A nearby match can indicate a tissue boundary or registration sensitivity. It cannot establish which tissue label is anatomically correct.
