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

## MRI and registration

Checked November 2026 for `notebooks/02_mri_pseudo_labels.ipynb`.

- **Ex vivo MRI** (`Derivatives/<subject>/MRI/exvivo/`): `raw/` has T1, T2 and B1 maps, a 0.2 mm `TSE_anatomical` on its own grid, and the brain mask (`brain_mask` for Vlad, `mask` for Moe; identical to `proc/data/nodif_brain_mask`); `proc/data_lowershell.DTI_SSFP/` has the DTI maps (`dti_FA`, `dti_MD`, `dti_V1`, ...). All except the TSE share one 0.4 mm grid (Vlad 170 x 200 x 112, Moe 160 x 200 x 112).
- **Registration** (`Enface/Registration/`, run with the lab's `psoct_wrapper`; Vlad's `command_log.txt`): FLIRT + FNIRT between MRI FA and PS-OCT retardance masked by cross-polarization, with warps both ways (`MRI_to_PSOCT_warpfield`, 2.6-2.9 GB, on the slide-deck grid; `PSOCT_to_MRI_warpfield`). `rel_mat.json` / `abs_mat.json` hold each section's in-plane transform into the slide deck.
- **Vlad's** registration FA is the MRI `dti_FA`, so its `*_in_MRI` volumes are on the MRI grid. **Moe's** is `dtifit_FA`: the same image scaled ~0.75x on a padded 346 x 346 x 112 grid with a placeholder origin. Moe's MRI voxel (i, j, k) is registration voxel (i + 90, j + 70, k) (cross-correlation 0.97); the NIfTI headers do not encode this.
- **Slide deck vs section files:** a deck slice is its `Registration/<contrast>/lowres/` section after the section's `abs_mat` transform (shifted ~2 mm for Vlad slice 137), so labels warped into deck space line up with the `*_slide_deck.nii.gz` volumes, not directly with the section files.
- **No ready-made tissue segmentation** exists for the PS-OCT: a search of the project storage found only the MRI brain masks and `Ret_masked_by_Cro_slide_deck`, whose nonzero voxels include the embedding gel (36% of them lie outside Vlad's brain mask).

## Values

Moe and Vlad were processed with different versions of the lab's MATLAB pipeline ([cmc_ps-oct_processing](https://github.com/mesoscale-connectivity/cmc_ps-oct_processing), `matlab_processing/`): Moe matches `PMSDOCT_2024_FCN.m`, Vlad matches `PSOCT_2025_FCN.m` (its stitcher is `MStitchFCN_Vlad.m`). Several differences below follow from that; the version assignment itself is inferred.

- **Reflectivity and cross-polarization** are uncalibrated dB (`10*log10` of the channel powers), averaged over the top ~160-185 depth pixels above a noise threshold. The offset depends on per-run settings, so absolute dB thresholds may not transfer between subjects.
- **Raw sections contain NaN** outside the imaged tiles (24.6% of Moe slice 160 cross-polarization); registered NIfTI files have none. The MATLAB pipeline itself writes `NoiseCut/2`, not NaN, for empty pixels, so the NaN comes from a later stage not in that repository. `brainseg` accepts NaN: those pixels are never used for training and are predicted as background (0).
- **Orientation: the fiber angle is half the phase.** `TEnAO` is complex128 and stores the doubled angle (the pipeline averages `exp(2i*theta)`-type phasors; `MStitchFCN_mod.m` and `Ori2RBG.m` use `angle(TEnAO)/2`). Use `brainseg.preprocess.orientation()`, which returns radians in [-pi/2, pi/2]. The registered orientation maps span [-pi, pi], so they appear to store the doubled angle as well (inferred). Between pipeline versions the phasor's conjugation order flips and 2025 adds a 42 degree offset, so Moe and Vlad orientations may be mirrored or offset relative to each other; check on a known tract before comparing them. `brainseg.io.read_array` rejects complex input rather than silently dropping the imaginary part.
- **Retardance units differ between subjects.** Moe ranges 0-70: the 2024 pipeline saves degrees (0-90). Vlad ranges 0-1.5: the 2025 pipeline saves a complex value whose phase was later taken in radians (max pi/2). Multiply Vlad by 180/pi to compare in degrees. The current methods normalize each image to its own 1st/99th percentiles, which is scale-invariant, so this only matters for fixed thresholds or pooled raw values.
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
