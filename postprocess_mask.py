#!/usr/bin/env python3
"""
Post-process segmentation masks to remove small disconnected objects.

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
    img = sitk.ReadImage(str(args.input))

    print(f"  Size: {img.GetSize()}")
    print(f"  Spacing: {img.GetSpacing()}")
    print(f"  Origin: {img.GetOrigin()}")

    # Binary threshold to get mask of all non-zero voxels
    binary = sitk.BinaryThreshold(img, lowerThreshold=1, upperThreshold=255)

    # Connected component labeling
    labeled = sitk.ConnectedComponent(binary, True)  # True = fully connected (26-connectivity)

    # Relabel components by size (largest = 1, second = 2, etc.)
    relabeled = sitk.RelabelComponent(labeled, sortByObjectSize=True)

    # Get stats
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(relabeled)

    labels = stats.GetLabels()
    print(f"  Found {len(labels)} connected components")

    if len(labels) > 0:
        print("  Component sizes (sorted by size):")
        for i, label in enumerate(labels):
            size = stats.GetNumberOfPixels(label)
            marker = " <-- KEEP" if i < args.keep else " <-- REMOVE"
            print(f"    #{i+1} (label {label}): {size:,} voxels{marker}")

    # Create mask to keep only the N largest components (labels 1 to N)
    if len(labels) <= args.keep:
        print(f"  Nothing to remove, copying input to output")
        sitk.WriteImage(img, str(args.output))
    else:
        # Keep labels 1 through args.keep (since relabeled by size, 1=largest)
        keep_mask = sitk.BinaryThreshold(relabeled,
                                          lowerThreshold=1,
                                          upperThreshold=args.keep)

        # Mask the original image - this preserves original label values
        output = sitk.Mask(img, keep_mask)

        print(f"\nSaving: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(output, str(args.output))

    print("Done!")


if __name__ == "__main__":
    main()
