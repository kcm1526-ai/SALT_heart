#!/usr/bin/env python3
"""
Post-process segmentation masks to remove small disconnected objects.

Uses SimpleITK's native connected component analysis - no numpy array conversion
that could mess up axis ordering.

Usage:
    python postprocess_mask.py --input mask.nii.gz --output mask_clean.nii.gz --keep 2
"""

import argparse
from pathlib import Path
import SimpleITK as sitk


def main():
    parser = argparse.ArgumentParser(
        description="Remove small disconnected objects from segmentation mask"
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input mask")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output mask")
    parser.add_argument("--keep", "-k", type=int, default=1, help="Keep N largest (default: 1)")

    args = parser.parse_args()

    print(f"Loading: {args.input}")

    # Load mask
    img = sitk.ReadImage(str(args.input))

    print(f"  Size: {img.GetSize()}")
    print(f"  Spacing: {img.GetSpacing()}")
    print(f"  Origin: {img.GetOrigin()}")

    # Create binary mask (any non-zero value)
    binary = sitk.BinaryThreshold(img, lowerThreshold=1, upperThreshold=255)

    # Connected component labeling (native SimpleITK - no numpy conversion)
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)  # 26-connectivity in 3D
    labeled = cc_filter.Execute(binary)

    num_components = cc_filter.GetObjectCount()
    print(f"  Found {num_components} connected components")

    if num_components <= args.keep:
        print(f"  Nothing to remove, saving as-is")
        sitk.WriteImage(img, str(args.output))
        return

    # Get size of each component using LabelShapeStatisticsImageFilter
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(labeled)

    # Collect component sizes
    component_sizes = []
    for label in stats.GetLabels():
        size = stats.GetNumberOfPixels(label)
        component_sizes.append((label, size))

    # Sort by size descending
    component_sizes.sort(key=lambda x: x[1], reverse=True)

    print("  Component sizes:")
    for i, (label, size) in enumerate(component_sizes):
        marker = " <-- KEEP" if i < args.keep else " <-- REMOVE"
        print(f"    #{i+1}: {size:,} voxels{marker}")

    # Get labels to keep
    keep_labels = [label for label, size in component_sizes[:args.keep]]

    # Create output mask - start with zeros, same properties as input
    output = sitk.Image(img.GetSize(), img.GetPixelID())
    output.CopyInformation(img)

    # For each kept component, copy the original values
    for keep_label in keep_labels:
        # Create mask for this component
        component_mask = sitk.BinaryThreshold(labeled,
                                               lowerThreshold=keep_label,
                                               upperThreshold=keep_label)
        # Copy original values where mask is 1
        output = sitk.Add(output, sitk.Mask(img, component_mask))

    # Save
    print(f"\nSaving: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output, str(args.output))

    # Count final voxels
    final_stats = sitk.LabelShapeStatisticsImageFilter()
    final_binary = sitk.BinaryThreshold(output, lowerThreshold=1, upperThreshold=255)
    final_cc = sitk.ConnectedComponentImageFilter()
    final_labeled = final_cc.Execute(final_binary)
    final_stats.Execute(final_labeled)

    total_voxels = sum(final_stats.GetNumberOfPixels(l) for l in final_stats.GetLabels())
    print(f"  Final non-zero voxels: {total_voxels:,}")
    print("Done!")


if __name__ == "__main__":
    main()
