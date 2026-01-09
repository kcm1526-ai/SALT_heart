#!/usr/bin/env python3
"""
Post-process segmentation masks to remove small disconnected objects.

Uses connected component analysis to keep only the largest objects.

Usage:
    # Keep only the largest object:
    python postprocess_mask.py --input mask.nii.gz --output mask_clean.nii.gz

    # Keep largest 2 objects:
    python postprocess_mask.py --input mask.nii.gz --output mask_clean.nii.gz --keep 2

    # Keep objects with at least 1000 voxels:
    python postprocess_mask.py --input mask.nii.gz --output mask_clean.nii.gz --min-size 1000

    # Process each label separately (for multi-label masks):
    python postprocess_mask.py --input mask.nii.gz --output mask_clean.nii.gz --per-label
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage


def get_connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Find connected components in a binary mask.

    Args:
        mask: Binary mask (0 and non-zero values)

    Returns:
        labeled_array: Array where each component has a unique label
        num_components: Number of components found
    """
    # Use 26-connectivity (3D diagonal connections)
    structure = ndimage.generate_binary_structure(3, 3)
    labeled_array, num_components = ndimage.label(mask > 0, structure=structure)
    return labeled_array, num_components


def get_component_sizes(labeled_array: np.ndarray, num_components: int) -> dict:
    """
    Get the size (voxel count) of each connected component.

    Args:
        labeled_array: Array from ndimage.label
        num_components: Number of components

    Returns:
        Dict mapping component label to size
    """
    sizes = {}
    for i in range(1, num_components + 1):
        sizes[i] = np.sum(labeled_array == i)
    return sizes


def keep_largest_components(
    mask: np.ndarray,
    keep_n: int = 1,
    min_size: Optional[int] = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Keep only the largest connected components in a binary mask.

    Args:
        mask: Input mask (can be binary or have a single non-zero value)
        keep_n: Number of largest components to keep
        min_size: Minimum size (voxels) to keep a component (overrides keep_n)
        verbose: Print component info

    Returns:
        Cleaned mask with only largest components
    """
    if np.sum(mask > 0) == 0:
        return mask

    # Get connected components
    labeled_array, num_components = get_connected_components(mask)

    if verbose:
        print(f"  Found {num_components} connected components")

    if num_components <= 1:
        return mask

    # Get component sizes
    sizes = get_component_sizes(labeled_array, num_components)

    # Sort by size (descending)
    sorted_components = sorted(sizes.items(), key=lambda x: x[1], reverse=True)

    if verbose:
        print("  Component sizes:")
        for label, size in sorted_components[:10]:  # Show top 10
            print(f"    Component {label}: {size:,} voxels")
        if len(sorted_components) > 10:
            print(f"    ... and {len(sorted_components) - 10} more")

    # Determine which components to keep
    if min_size is not None:
        # Keep all components >= min_size
        keep_labels = [label for label, size in sorted_components if size >= min_size]
        if verbose:
            print(f"  Keeping {len(keep_labels)} components with >= {min_size} voxels")
    else:
        # Keep top N largest
        keep_labels = [label for label, size in sorted_components[:keep_n]]
        if verbose:
            print(f"  Keeping {len(keep_labels)} largest components")

    # Create output mask
    output_mask = np.zeros_like(mask)
    original_value = mask[mask > 0][0] if np.any(mask > 0) else 1

    for label in keep_labels:
        output_mask[labeled_array == label] = original_value

    removed_voxels = np.sum(mask > 0) - np.sum(output_mask > 0)
    if verbose:
        print(f"  Removed {removed_voxels:,} voxels ({100*removed_voxels/np.sum(mask > 0):.1f}%)")

    return output_mask


def postprocess_multilabel_mask(
    mask: np.ndarray,
    keep_n: int = 1,
    min_size: Optional[int] = None,
    per_label: bool = False,
    verbose: bool = True
) -> np.ndarray:
    """
    Post-process a multi-label segmentation mask.

    Args:
        mask: Input mask with multiple label values
        keep_n: Number of largest components to keep
        min_size: Minimum size to keep
        per_label: If True, process each label separately
        verbose: Print info

    Returns:
        Cleaned mask
    """
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background

    if verbose:
        print(f"Found {len(unique_labels)} unique labels: {unique_labels.tolist()}")

    if per_label:
        # Process each label separately
        output_mask = np.zeros_like(mask)

        for label in unique_labels:
            if verbose:
                print(f"\nProcessing label {label}:")

            binary_mask = (mask == label).astype(np.uint8)
            cleaned = keep_largest_components(
                binary_mask,
                keep_n=keep_n,
                min_size=min_size,
                verbose=verbose
            )
            output_mask[cleaned > 0] = label

        return output_mask
    else:
        # Process all labels together as one mask
        if verbose:
            print("\nProcessing all labels together:")

        binary_mask = (mask > 0).astype(np.uint8)
        labeled_array, num_components = get_connected_components(binary_mask)

        if verbose:
            print(f"  Found {num_components} connected components")

        if num_components <= 1:
            return mask

        # Get component sizes
        sizes = get_component_sizes(labeled_array, num_components)
        sorted_components = sorted(sizes.items(), key=lambda x: x[1], reverse=True)

        if verbose:
            print("  Component sizes:")
            for label, size in sorted_components[:10]:
                print(f"    Component {label}: {size:,} voxels")

        # Determine which to keep
        if min_size is not None:
            keep_labels = [label for label, size in sorted_components if size >= min_size]
        else:
            keep_labels = [label for label, size in sorted_components[:keep_n]]

        if verbose:
            print(f"  Keeping {len(keep_labels)} components")

        # Create output - preserve original label values
        output_mask = np.zeros_like(mask)
        for comp_label in keep_labels:
            component_mask = labeled_array == comp_label
            output_mask[component_mask] = mask[component_mask]

        removed_voxels = np.sum(mask > 0) - np.sum(output_mask > 0)
        if verbose:
            print(f"  Removed {removed_voxels:,} voxels")

        return output_mask


def main():
    parser = argparse.ArgumentParser(
        description="Remove small disconnected objects from segmentation masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Keep only the largest connected object:
  python postprocess_mask.py -i mask.nii.gz -o mask_clean.nii.gz

  # Keep the 2 largest objects:
  python postprocess_mask.py -i mask.nii.gz -o mask_clean.nii.gz --keep 2

  # Keep objects with at least 5000 voxels:
  python postprocess_mask.py -i mask.nii.gz -o mask_clean.nii.gz --min-size 5000

  # Process each label value separately:
  python postprocess_mask.py -i mask.nii.gz -o mask_clean.nii.gz --per-label

  # Combine: keep largest per label, minimum 1000 voxels:
  python postprocess_mask.py -i mask.nii.gz -o mask_clean.nii.gz --per-label --min-size 1000
        """
    )

    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="Input mask file (NIfTI)"
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Output mask file (NIfTI)"
    )
    parser.add_argument(
        "--keep", "-k", type=int, default=1,
        help="Number of largest components to keep (default: 1)"
    )
    parser.add_argument(
        "--min-size", "-m", type=int, default=None,
        help="Minimum component size in voxels (overrides --keep)"
    )
    parser.add_argument(
        "--per-label", "-p", action="store_true",
        help="Process each label value separately"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress output"
    )

    args = parser.parse_args()

    # Load mask
    if not args.quiet:
        print(f"Loading: {args.input}")

    img = nib.load(args.input)
    mask = img.get_fdata().astype(np.uint8)

    if not args.quiet:
        print(f"Shape: {mask.shape}")
        print(f"Total non-zero voxels: {np.sum(mask > 0):,}")

    # Process
    cleaned_mask = postprocess_multilabel_mask(
        mask,
        keep_n=args.keep,
        min_size=args.min_size,
        per_label=args.per_label,
        verbose=not args.quiet
    )

    # Save
    output_img = nib.Nifti1Image(
        cleaned_mask.astype(np.uint8),
        affine=img.affine,
        header=img.header
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output_img, args.output)

    if not args.quiet:
        print(f"\nSaved: {args.output}")
        print(f"Final non-zero voxels: {np.sum(cleaned_mask > 0):,}")


if __name__ == "__main__":
    main()
