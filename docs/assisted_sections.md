# MRI-assisted native sections

The optical mask can miss dark regions inside the registered brain. On section 110, the workflow fills eligible pixels from each animal's MRI-derived tissue labels. Use the source map to distinguish optical estimates, MRI estimates and unresolved tissue.

## Run a section

Install the project with `pip install -e ".[mri]"`. Notebook 02 produces `<subject>_mri_labels.nii.gz` in the original MRI grid. Use those semantic labels and the matching subject's CMC registration files. The native reflectivity and retardance inputs must be the matching 10× downsampled registered NIfTI sections in the two `lowres/` folders. Their affines locate the section on the optical slide deck.

Set `REG` to the subject's `PS-OCT/Enface/Registration` directory, `MRI_LABELS` to its MRI labels, and `REF_SECTION` and `RET_SECTION` to the two section files. For Vlad section 110 run

```bash
brainseg assist-section \
  --reflectivity "$REF_SECTION" --retardance "$RET_SECTION" \
  --deck "$REG/Ref_slide_deck.nii.gz" --registered-ref "$REG/Ref_slide_deck_in_MRI.nii.gz" \
  --warp "$REG/PSOCT_to_MRI_warpfield.nii.gz" --mri-labels "$MRI_LABELS" \
  --floor 32.5 --output outputs/vlad110
```

For Moe, point all four variables at Moe's inputs and replace the last line with `--offset 90 70 0 --output outputs/moe110`. The offset locates Moe's MRI inside the padded registration grid. Vlad's value of 32.5 is the measured reflectivity floor for this section.

## Reading the output

- `tissue.nii.gz` stores the native section's tissue IDs (0 background, 1 gray matter, 2 white matter, 3 possible fluid, 255 uncertain).
- `source.nii.gz` records 0 background, 1 optical estimate, 2 MRI fill and 255 uncertain.
- `report.json` records source counts, the selected floor and offset, and the correlation between smoothed native reflectivity and the registered reflectivity sampled back onto the section. The correlation checks geometric consistency. Tissue accuracy requires an independent reference.

The command runs the existing Otsu method on reflectivity and retardance. It maps each optical pixel through the slide-deck geometry, numerically inverts the lab's relative FNIRT displacement in FSL scaled millimetres, and samples categorical MRI labels by nearest neighbor. Within MRI support, optical background and pixels at the specified floor become eligible for filling. An uncertain MRI label leaves an eligible pixel uncertain. Other optical assignments are preserved, including optical tissue beyond the MRI boundary.

| Section | Optical estimates | MRI fills | Uncertain | Reflectivity correlation |
| --- | ---: | ---: | ---: | ---: |
| Moe 110, coronal | 314,357 | 48,434 | 223 | 0.896 |
| Vlad 110, sagittal | 287,143 | 315,807 | 38,244 | 0.970 |

The output labels on MRI-filled pixels come directly from the MRI reference. Exclude those pixels when measuring independent optical agreement with that reference.
