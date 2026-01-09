#!/usr/bin/env python3
"""
Post-process segmentation masks to remove small disconnected objects.

Uses nibabel (same as inference script) to avoid coordinate system mismatches.

Usage:
    python postprocess_mask.py --input mask.nii.gz --output mask_clean.nii.gz --keep 2
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def main():
    parser = argparse.ArgumentParser(
        description="Remove small disconnected objects from segmentation mask"
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input mask")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output mask")
    parser.add_argument("--keep", "-k", type=int, default=1, help="Keep N largest (default: 1)")

    args = parser.parse_args()

    print(f"Loading: {args.input}")

    # Load with nibabel (same library used to save the mask in inference)
    img = nib.load(args.input)
    data = img.get_fdata()

    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    print(f"  Non-zero voxels: {np.sum(data > 0):,}")

    # Find connected components
    structure = ndimage.generate_binary_structure(3, 3)  # 26-connectivity
    labeled, num_components = ndimage.label(data > 0, structure=structure)

    print(f"  Found {num_components} connected components")

    if num_components <= args.keep:
        print("  Nothing to remove, copying as-is")
        nib.save(img, args.output)
        print(f"Saved: {args.output}")
        return

    # Get component sizes
    component_sizes = ndimage.sum(data > 0, labeled, range(1, num_components + 1))

    # Create list of (component_id, size) and sort by size descending
    sizes_list = [(i + 1, size) for i, size in enumerate(component_sizes)]
    sizes_list.sort(key=lambda x: x[1], reverse=True)

    print("  Component sizes:")
    for i, (comp_id, size) in enumerate(sizes_list):
        marker = " <-- KEEP" if i < args.keep else " <-- REMOVE"
        print(f"    #{i+1}: {int(size):,} voxels{marker}")

    # Create mask: 1 where we keep, 0 where we remove
    keep_ids = set(comp_id for comp_id, _ in sizes_list[:args.keep])
    keep_mask = np.isin(labeled, list(keep_ids))

    # Zero out removed voxels (keep original values where mask is True)
    output_data = data * keep_mask

    removed = np.sum(data > 0) - np.sum(output_data > 0)
    print(f"  Removed {int(removed):,} voxels")

    # Save with EXACT same affine and header
    output_img = nib.Nifti1Image(
        output_data.astype(img.get_data_dtype()),
        affine=img.affine,
        header=img.header
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output_img, args.output)

    print(f"Saved: {args.output}")
    print(f"  Final non-zero voxels: {np.sum(output_data > 0):,}")
    print("Done!")


if __name__ == "__main__":
    main()
