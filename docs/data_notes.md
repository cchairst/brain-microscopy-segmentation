# PS-OCT data notes

Findings from `notebooks/01_explore_contrasts.ipynb` on Moe and Vlad (`Derivatives/<subject>/PS-OCT/Enface/` in the lab project). Checked October 2026.

## Files

| Folder | Contents | Moe | Vlad |
| --- | --- | --- | --- |
| `<Contrast>/Slice_NNN_En*.mat` | Raw en-face section, one variable (`TEnRef`, `TEnCr`, `TEnR`, `TEnAO`), MATLAB v5 | 8731 x 11000, ~0.4-1.1 GB each | not checked |
| `Registration/<Contrast>/Slice_NNN_En*_hdr.nii.gz` | Registered section, full resolution | ~170-275 MB each | |
| `Registration/<Contrast>/lowres/` | Registered section, 10x downsampled, 0.055 mm in-plane | 874 x 1100 | 820 x 1270 |
| `Registration/*_slide_deck.nii.gz` | All registered sections stacked into a volume | 0.4-0.7 GB per contrast | |
| `Registration/*_slide_deck_in_MRI.nii.gz` | Slide deck resampled to MRI space, 0.4 mm, RAS | 346 x 346 x 112 | 170 x 200 x 112 |

`Registration/slidedeck_slice_mapping.json` maps volume index to section: the stack is in reverse order (index 0 is Moe's slice 327).

## Values

- **Raw sections contain NaN** outside the imaged tiles (24.6% of Moe slice 160 cross-polarization). Registered NIfTI files have none. `brainseg.io.read_array` rejects non-finite input, so raw sections need masking before use in the benchmark.
- **Raw orientation is complex-valued.** `TEnAO` is complex128; its phase (`np.angle`) matches the registered orientation maps in radians [-pi, pi]. Casting it to float silently drops the imaginary part.
- **Retardance units differ between subjects.** Moe ranges 0-70 (consistent with degrees), Vlad 0-1.5 (consistent with radians); 1.1 rad is about 63 degrees. Convert before pooling subjects or sharing thresholds. Units are inferred from ranges, not documented.
- Raw and registered sections show the same image: raw retardance downsampled 10x correlates 0.98 with the registered low-res file after transposing (raw is rows x columns, NIfTI is columns x rows).

## Contrasts

Visual assessment of 10 evenly spaced sections per subject; not a quantitative result.

- **Retardance** shows the clearest gray/white matter boundary in both subjects.
- **Cross-polarization** separates tissue from background in Moe; in Vlad the surrounding gel is mid-gray with a visible tile pattern.
- **Reflectivity** shows little gray/white contrast at low resolution; full resolution shows fiber structure. Useful for voids.
- **Orientation** has strong tile-seam stripes outside tissue and encodes fiber direction rather than tissue class.

## Artifacts seen

- End sections are mostly empty (Moe slices 1 and 327, Vlad slices 1 and 245).
- Large voids: Moe slice 292, Vlad slice 109.
- Reflectivity scale differs from neighbors: Vlad slice 82.
- Corner wedges at the image edge: Moe slices 185 and 220.
