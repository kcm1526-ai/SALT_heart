#!/usr/bin/env python3
"""
Heart Segmentation Visualization Tool

Simple visualization of CT images with heart segmentation mask overlay.

Usage:
    python viz.py --image image.nii.gz --mask mask.nii.gz
    python viz.py --image image.nii.gz --mask mask.nii.gz --save output.png

Author: Generated for SALT heart segmentation
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
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


def load_nifti(path: Path) -> np.ndarray:
    """Load NIfTI file and return data."""
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


def interactive_viewer(image: np.ndarray, mask: np.ndarray):
    """Simple interactive slice viewer."""
    # Normalize image
    image_norm = normalize_image(image)
    cmap = create_colormap()

    num_slices = image.shape[2]
    current_slice = num_slices // 2

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.15)

    # Initial display
    img_slice = np.rot90(image_norm[:, :, current_slice])
    mask_slice = np.rot90(mask[:, :, current_slice])

    im = ax.imshow(img_slice, cmap='gray')
    mask_im = ax.imshow(np.ma.masked_where(mask_slice == 0, mask_slice),
                        cmap=cmap, alpha=0.5, vmin=0, vmax=8)
    ax.set_title(f'Slice {current_slice}/{num_slices-1}')
    ax.axis('off')

    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, num_slices - 1,
                    valinit=current_slice, valstep=1)

    def update(val):
        idx = int(slider.val)
        img_slice = np.rot90(image_norm[:, :, idx])
        mask_slice = np.rot90(mask[:, :, idx])

        im.set_data(img_slice)
        mask_im.set_data(np.ma.masked_where(mask_slice == 0, mask_slice))
        ax.set_title(f'Slice {idx}/{num_slices-1}')
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Scroll wheel support
    def on_scroll(event):
        if event.button == 'up':
            new_val = min(slider.val + 1, num_slices - 1)
        else:
            new_val = max(slider.val - 1, 0)
        slider.set_val(new_val)

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # Add legend
    unique_labels = np.unique(mask).astype(int)
    legend_text = []
    for label in unique_labels:
        if label > 0 and label in HEART_LABELS:
            color = HEART_COLORS[label][:3]
            legend_text.append(f"{HEART_LABELS[label]}")

    if legend_text:
        fig.text(0.02, 0.98, "Structures:\n" + "\n".join(legend_text),
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.show()


def save_image(image: np.ndarray, mask: np.ndarray, output_path: Path,
               slice_idx: int = None):
    """Save a single slice or montage."""
    image_norm = normalize_image(image)
    cmap = create_colormap()

    num_slices = image.shape[2]

    if slice_idx is None:
        # Save montage
        n_show = 16
        indices = np.linspace(num_slices * 0.2, num_slices * 0.8, n_show, dtype=int)

        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()

        for i, idx in enumerate(indices):
            img_slice = np.rot90(image_norm[:, :, idx])
            mask_slice = np.rot90(mask[:, :, idx])

            axes[i].imshow(img_slice, cmap='gray')
            axes[i].imshow(np.ma.masked_where(mask_slice == 0, mask_slice),
                          cmap=cmap, alpha=0.5, vmin=0, vmax=8)
            axes[i].set_title(f'Slice {idx}')
            axes[i].axis('off')

        plt.tight_layout()
    else:
        # Save single slice
        fig, ax = plt.subplots(figsize=(10, 10))

        img_slice = np.rot90(image_norm[:, :, slice_idx])
        mask_slice = np.rot90(mask[:, :, slice_idx])

        ax.imshow(img_slice, cmap='gray')
        ax.imshow(np.ma.masked_where(mask_slice == 0, mask_slice),
                  cmap=cmap, alpha=0.5, vmin=0, vmax=8)
        ax.set_title(f'Slice {slice_idx}')
        ax.axis('off')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize heart segmentation')
    parser.add_argument('--image', '-i', type=Path, required=True,
                        help='Input CT image (NIfTI)')
    parser.add_argument('--mask', '-m', type=Path, required=True,
                        help='Segmentation mask (NIfTI)')
    parser.add_argument('--save', '-s', type=Path, default=None,
                        help='Save to file instead of interactive display')
    parser.add_argument('--slice', type=int, default=None,
                        help='Specific slice index (default: middle slice for single, montage for save)')

    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.mask.exists():
        raise FileNotFoundError(f"Mask not found: {args.mask}")

    print(f"Loading image: {args.image}")
    image = load_nifti(args.image)
    print(f"Loading mask: {args.mask}")
    mask = load_nifti(args.mask)

    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Unique mask values: {np.unique(mask).astype(int)}")

    if args.save:
        save_image(image, mask, args.save, args.slice)
    else:
        interactive_viewer(image, mask)


if __name__ == '__main__':
    main()
