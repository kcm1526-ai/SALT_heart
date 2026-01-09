#!/usr/bin/env python3
"""
Post-process segmentation masks to remove small disconnected objects.

ONLY deletes small objects - does NOT change position, orientation, or any spatial info.

Usage:
    # Keep only the largest object:
    python postprocess_mask.py --input mask.nii.gz --output mask_clean.nii.gz

    # Keep largest 2 objects:
    python postprocess_mask.py --input mask.nii.gz --output mask_clean.nii.gz --keep 2
"""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


def remove_small_objects(mask_data: np.ndarray, keep_n: int = 1) -> np.ndarray:
    """
    Remove small disconnected objects, keeping only the N largest.

    Only zeros out voxels - does NOT change anything else.
    """
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

    # Zero out small components in-place
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

    # Load with SimpleITK (preserves all spatial metadata perfectly)
    img = sitk.ReadImage(str(args.input))

    # Get array (SimpleITK uses zyx order)
    mask_data = sitk.GetArrayFromImage(img)

    print(f"  Shape: {mask_data.shape}")
    print(f"  Spacing: {img.GetSpacing()}")
    print(f"  Origin: {img.GetOrigin()}")
    print(f"  Direction: {img.GetDirection()}")
    print(f"  Non-zero voxels: {np.sum(mask_data > 0):,}")
    print(f"  Unique values: {np.unique(mask_data).tolist()}")

    # Remove small objects
    print(f"\nRemoving small objects (keeping {args.keep} largest)...")
    cleaned = remove_small_objects(mask_data, keep_n=args.keep)

    # Convert back to SimpleITK image
    output_img = sitk.GetImageFromArray(cleaned)

    # Copy ALL spatial metadata from original
    output_img.CopyInformation(img)

    # Save
    print(f"\nSaving: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output_img, str(args.output))

    print(f"  Final non-zero voxels: {np.sum(cleaned > 0):,}")
    print("Done!")


if __name__ == "__main__":
    main()
