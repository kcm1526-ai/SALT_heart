#!/usr/bin/env python3
"""
Heart Segmentation Visualization Tool

Simple visualization of CT images with heart segmentation mask overlay.
Shows all three views: Axial, Coronal, and Sagittal.

Usage:
    python viz.py --image image.nii.gz --mask mask.nii.gz
    python viz.py --image /path/to/dicom_folder --mask mask.nii.gz
    python viz.py --image image.nii.gz --mask mask.nii.gz --save output.png

Author: Generated for SALT heart segmentation
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Slider


# Color map for heart structures
HEART_COLORS = {
    0: (0, 0, 0, 0),           # Background (transparent)
    1: (1, 0, 0, 0.6),         # heart_myocardium - Red
    2: (0, 0, 1, 0.6),         # heart_atrium_left - Blue
    3: (0, 1, 0, 0.6),         # heart_ventricle_left - Green
    4: (1, 1, 0, 0.6),         # heart_atrium_right - Yellow
    5: (1, 0, 1, 0.6),         # heart_ventricle_right - Magenta
    6: (0, 1, 1, 0.6),         # pericardium - Cyan
    7: (1, 0.5, 0, 0.6),       # aorta - Orange
    8: (0.5, 0, 1, 0.6),       # pulmonary artery - Purple
}

HEART_LABELS = {
    1: "Heart Myocardium",
    2: "Left Atrium",
    3: "Left Ventricle",
    4: "Right Atrium",
    5: "Right Ventricle",
    6: "Pericardium",
    7: "Aorta",
    8: "Pulmonary Artery",
}


def load_dicom_folder(dicom_path: Path) -> np.ndarray:
    """Load DICOM series from folder."""
    reader = sitk.ImageSeriesReader()

    # Try to get series IDs
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_path))

    if len(series_ids) == 0:
        # Try finding .dcm files directly
        dcm_files = list(dicom_path.glob("*.dcm")) + list(dicom_path.glob("*.DCM"))
        if len(dcm_files) == 0:
            raise ValueError(f"No DICOM files found in {dicom_path}")
        dcm_files = sorted(dcm_files, key=lambda x: x.name)
        dicom_names = [str(f) for f in dcm_files]
    else:
        dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            str(dicom_path), series_ids[0]
        )

    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # Convert to numpy array
    data = sitk.GetArrayFromImage(image)
    # SimpleITK returns (Z, Y, X), transpose to (X, Y, Z) for consistency
    data = np.transpose(data, (2, 1, 0))
    return data


def load_image(path: Path) -> np.ndarray:
    """Load image from NIfTI file or DICOM folder."""
    if path.is_dir():
        # DICOM folder
        print(f"Loading DICOM folder: {path}")
        return load_dicom_folder(path)
    elif path.suffix.lower() in ['.dcm']:
        # Single DICOM file - load parent folder
        print(f"Loading DICOM from parent folder: {path.parent}")
        return load_dicom_folder(path.parent)
    else:
        # NIfTI file
        print(f"Loading NIfTI file: {path}")
        img = nib.load(path)
        img_canonical = nib.as_closest_canonical(img)
        return img_canonical.get_fdata()


def normalize_image(image: np.ndarray, window_center: int = 40, window_width: int = 400) -> np.ndarray:
    """Apply CT windowing."""
    min_val = window_center - window_width // 2
    max_val = window_center + window_width // 2
    image = np.clip(image, min_val, max_val)
    return (image - min_val) / (max_val - min_val)


def create_colormap() -> ListedColormap:
    """Create colormap for mask."""
    colors = [HEART_COLORS.get(i, (0.5, 0.5, 0.5, 0.5)) for i in range(9)]
    return ListedColormap(colors)


def get_slice_axial(data, idx):
    """Get axial slice (top-down view)."""
    return np.rot90(data[:, :, idx])


def get_slice_coronal(data, idx):
    """Get coronal slice (front view)."""
    return np.rot90(data[:, idx, :])


def get_slice_sagittal(data, idx):
    """Get sagittal slice (side view)."""
    return np.rot90(data[idx, :, :])


def interactive_viewer(image: np.ndarray, mask: np.ndarray):
    """Interactive viewer with all three views."""
    # Normalize image
    image_norm = normalize_image(image)
    cmap = create_colormap()

    # Get dimensions
    nx, ny, nz = image.shape

    # Initial slice positions (middle of each axis)
    axial_idx = nz // 2
    coronal_idx = ny // 2
    sagittal_idx = nx // 2

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    plt.subplots_adjust(bottom=0.25)

    # Axial view
    ax_slice = get_slice_axial(image_norm, axial_idx)
    ax_mask = get_slice_axial(mask, axial_idx)
    im_axial = axes[0].imshow(ax_slice, cmap='gray')
    mask_axial = axes[0].imshow(np.ma.masked_where(ax_mask == 0, ax_mask),
                                 cmap=cmap, alpha=0.5, vmin=0, vmax=8)
    axes[0].set_title(f'Axial (slice {axial_idx}/{nz-1})')
    axes[0].axis('off')

    # Coronal view
    cor_slice = get_slice_coronal(image_norm, coronal_idx)
    cor_mask = get_slice_coronal(mask, coronal_idx)
    im_coronal = axes[1].imshow(cor_slice, cmap='gray')
    mask_coronal = axes[1].imshow(np.ma.masked_where(cor_mask == 0, cor_mask),
                                   cmap=cmap, alpha=0.5, vmin=0, vmax=8)
    axes[1].set_title(f'Coronal (slice {coronal_idx}/{ny-1})')
    axes[1].axis('off')

    # Sagittal view
    sag_slice = get_slice_sagittal(image_norm, sagittal_idx)
    sag_mask = get_slice_sagittal(mask, sagittal_idx)
    im_sagittal = axes[2].imshow(sag_slice, cmap='gray')
    mask_sagittal = axes[2].imshow(np.ma.masked_where(sag_mask == 0, sag_mask),
                                    cmap=cmap, alpha=0.5, vmin=0, vmax=8)
    axes[2].set_title(f'Sagittal (slice {sagittal_idx}/{nx-1})')
    axes[2].axis('off')

    # Sliders
    ax_slider_axial = plt.axes([0.1, 0.15, 0.25, 0.03])
    ax_slider_coronal = plt.axes([0.4, 0.15, 0.25, 0.03])
    ax_slider_sagittal = plt.axes([0.7, 0.15, 0.25, 0.03])

    slider_axial = Slider(ax_slider_axial, 'Axial', 0, nz - 1, valinit=axial_idx, valstep=1)
    slider_coronal = Slider(ax_slider_coronal, 'Coronal', 0, ny - 1, valinit=coronal_idx, valstep=1)
    slider_sagittal = Slider(ax_slider_sagittal, 'Sagittal', 0, nx - 1, valinit=sagittal_idx, valstep=1)

    def update_axial(val):
        idx = int(slider_axial.val)
        ax_slice = get_slice_axial(image_norm, idx)
        ax_mask = get_slice_axial(mask, idx)
        im_axial.set_data(ax_slice)
        mask_axial.set_data(np.ma.masked_where(ax_mask == 0, ax_mask))
        axes[0].set_title(f'Axial (slice {idx}/{nz-1})')
        fig.canvas.draw_idle()

    def update_coronal(val):
        idx = int(slider_coronal.val)
        cor_slice = get_slice_coronal(image_norm, idx)
        cor_mask = get_slice_coronal(mask, idx)
        im_coronal.set_data(cor_slice)
        mask_coronal.set_data(np.ma.masked_where(cor_mask == 0, cor_mask))
        axes[1].set_title(f'Coronal (slice {idx}/{ny-1})')
        fig.canvas.draw_idle()

    def update_sagittal(val):
        idx = int(slider_sagittal.val)
        sag_slice = get_slice_sagittal(image_norm, idx)
        sag_mask = get_slice_sagittal(mask, idx)
        im_sagittal.set_data(sag_slice)
        mask_sagittal.set_data(np.ma.masked_where(sag_mask == 0, sag_mask))
        axes[2].set_title(f'Sagittal (slice {idx}/{nx-1})')
        fig.canvas.draw_idle()

    slider_axial.on_changed(update_axial)
    slider_coronal.on_changed(update_coronal)
    slider_sagittal.on_changed(update_sagittal)

    # Add legend
    unique_labels = np.unique(mask).astype(int)
    legend_text = []
    for label in unique_labels:
        if label > 0 and label in HEART_LABELS:
            legend_text.append(f"{HEART_LABELS[label]}")

    if legend_text:
        fig.text(0.02, 0.08, "Structures: " + ", ".join(legend_text),
                 fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.show()


def save_image(image: np.ndarray, mask: np.ndarray, output_path: Path, slice_idx: int = None):
    """Save visualization with all three views."""
    image_norm = normalize_image(image)
    cmap = create_colormap()

    nx, ny, nz = image.shape

    # Use middle slices if not specified
    axial_idx = nz // 2 if slice_idx is None else min(slice_idx, nz - 1)
    coronal_idx = ny // 2
    sagittal_idx = nx // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # Axial
    ax_slice = get_slice_axial(image_norm, axial_idx)
    ax_mask = get_slice_axial(mask, axial_idx)
    axes[0].imshow(ax_slice, cmap='gray')
    axes[0].imshow(np.ma.masked_where(ax_mask == 0, ax_mask), cmap=cmap, alpha=0.5, vmin=0, vmax=8)
    axes[0].set_title(f'Axial (slice {axial_idx})')
    axes[0].axis('off')

    # Coronal
    cor_slice = get_slice_coronal(image_norm, coronal_idx)
    cor_mask = get_slice_coronal(mask, coronal_idx)
    axes[1].imshow(cor_slice, cmap='gray')
    axes[1].imshow(np.ma.masked_where(cor_mask == 0, cor_mask), cmap=cmap, alpha=0.5, vmin=0, vmax=8)
    axes[1].set_title(f'Coronal (slice {coronal_idx})')
    axes[1].axis('off')

    # Sagittal
    sag_slice = get_slice_sagittal(image_norm, sagittal_idx)
    sag_mask = get_slice_sagittal(mask, sagittal_idx)
    axes[2].imshow(sag_slice, cmap='gray')
    axes[2].imshow(np.ma.masked_where(sag_mask == 0, sag_mask), cmap=cmap, alpha=0.5, vmin=0, vmax=8)
    axes[2].set_title(f'Sagittal (slice {sagittal_idx})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize heart segmentation (Axial, Coronal, Sagittal)')
    parser.add_argument('--image', '-i', type=Path, required=True,
                        help='Input CT image (NIfTI or DICOM folder)')
    parser.add_argument('--mask', '-m', type=Path, required=True,
                        help='Segmentation mask (NIfTI)')
    parser.add_argument('--save', '-s', type=Path, default=None,
                        help='Save to file instead of interactive display')
    parser.add_argument('--slice', type=int, default=None,
                        help='Axial slice index for saved image')

    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.mask.exists():
        raise FileNotFoundError(f"Mask not found: {args.mask}")

    print(f"Loading image: {args.image}")
    image = load_image(args.image)
    print(f"Loading mask: {args.mask}")
    mask = load_image(args.mask)

    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Unique mask values: {np.unique(mask).astype(int)}")

    if args.save:
        save_image(image, mask, args.save, args.slice)
    else:
        interactive_viewer(image, mask)


if __name__ == '__main__':
    main()
