"""Tissue segmentation and benchmarking for brain microscopy (PS-OCT first)."""

LABELS = {0: 'background', 1: 'gray_matter', 2: 'white_matter', 3: 'fluid_space'}
IGNORE = 255  # unannotated, uncertain tissue, or excluded artifact
