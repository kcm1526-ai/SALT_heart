#!/usr/bin/env python3
"""
Post-process segmentation masks to remove small disconnected objects.

ONLY deletes small objects - does NOT change position, affine, or any spatial info.

Usage:
    # Keep only the largest object:
    python postprocess_mask.py --input mask.nii.gz --output mask_clean.nii.gz

    # Keep largest 2 objects:
    python postprocess_mask.py --input mask.nii.gz --output mask_clean.nii.gz --keep 2
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def remove_small_objects(mask_data: np.ndarray, keep_n: int = 1) -> np.ndarray:
    """
    Remove small disconnected objects, keeping only the N largest.

    Only zeros out voxels - does NOT change anything else.

    Args:
        mask_data: Input mask array
        keep_n: Number of largest components to keep

    Returns:
        Same array with small objects set to 0
    """
    # Work on binary version to find components
    binary = mask_data > 0

    if np.sum(binary) == 0:
        print("  No non-zero voxels found")
        return mask_data

    # Find connected components (26-connectivity for 3D)
    structure = ndimage.generate_binary_structure(3, 3)
    labeled, num_components = ndimage.label(binary, structure=structure)

    print(f"  Found {num_components} connected components")

    if num_components <= keep_n:
        print(f"  Nothing to remove (keeping all {num_components})")
        return mask_data

    # Get size of each component
    component_sizes = []
    for i in range(1, num_components + 1):
        size = np.sum(labeled == i)
        component_sizes.append((i, size))

    # Sort by size descending
    component_sizes.sort(key=lambda x: x[1], reverse=True)

    print("  Component sizes:")
    for i, (comp_id, size) in enumerate(component_sizes):
        marker = " <-- KEEP" if i < keep_n else " <-- REMOVE"
        print(f"    #{i+1}: {size:,} voxels{marker}")

    # Get labels to keep
    keep_labels = set(comp_id for comp_id, _ in component_sizes[:keep_n])

    # Create output - copy original and zero out removed components
    output = mask_data.copy()

    for comp_id, size in component_sizes[keep_n:]:
        output[labeled == comp_id] = 0

    removed = np.sum(mask_data > 0) - np.sum(output > 0)
    print(f"  Removed {removed:,} voxels")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Remove small disconnected objects from segmentation mask"
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input mask")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output mask")
    parser.add_argument("--keep", "-k", type=int, default=1, help="Keep N largest (default: 1)")

    args = parser.parse_args()

    print(f"Loading: {args.input}")

    # Load NIfTI
    img = nib.load(args.input)

    # Get data as same dtype
    original_dtype = img.get_data_dtype()
    mask_data = np.asarray(img.dataobj)  # Preserves original dtype

    print(f"  Shape: {mask_data.shape}")
    print(f"  Dtype: {mask_data.dtype}")
    print(f"  Non-zero voxels: {np.sum(mask_data > 0):,}")
    print(f"  Unique values: {np.unique(mask_data).tolist()}")

    # Remove small objects
    print(f"\nRemoving small objects (keeping {args.keep} largest)...")
    cleaned = remove_small_objects(mask_data, keep_n=args.keep)

    # Save with EXACT same affine and header
    print(f"\nSaving: {args.output}")

    # Create new image with same spatial info
    output_img = nib.Nifti1Image(
        cleaned.astype(original_dtype),
        affine=img.affine.copy(),  # Copy to be safe
        header=img.header.copy()   # Copy to be safe
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output_img, args.output)

    print(f"  Final non-zero voxels: {np.sum(cleaned > 0):,}")
    print("Done!")


if __name__ == "__main__":
    main()
