#!/usr/bin/env python3
"""
Heart Segmentation Visualization Tool

Visualizes CT images with heart segmentation mask overlay.

Usage:
    python viz.py --image image.nii.gz --mask mask.nii.gz
    python viz.py --image image.nii.gz --mask mask.nii.gz --save output.png
    python viz.py --image image.nii.gz --mask mask.nii.gz --axis axial --slice 100

Author: Generated for SALT heart segmentation
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Slider, Button, RadioButtons


# Color map for heart structures (matching infer_heart.py output)
HEART_COLORS = {
    0: (0, 0, 0, 0),           # Background (transparent)
    1: (1, 0, 0, 0.6),         # heart_myocardium - Red
    2: (0, 0, 1, 0.6),         # heart_atrium_left - Blue
    3: (0, 1, 0, 0.6),         # heart_ventricle_left - Green
    4: (1, 1, 0, 0.6),         # heart_atrium_right - Yellow
    5: (1, 0, 1, 0.6),         # heart_ventricle_right - Magenta
    6: (0, 1, 1, 0.6),         # pericardium - Cyan
    7: (1, 0.5, 0, 0.6),       # aorta_thoracica_pass_pericardium - Orange
    8: (0.5, 0, 1, 0.6),       # pulmonary_artery_pass_pericardium - Purple
}

HEART_LABELS = {
    0: "Background",
    1: "Heart Myocardium",
    2: "Left Atrium",
    3: "Left Ventricle",
    4: "Right Atrium",
    5: "Right Ventricle",
    6: "Pericardium",
    7: "Aorta (Pericardium)",
    8: "Pulmonary Artery (Pericardium)",
}


def load_nifti(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """Load NIfTI file and return data with image object."""
    img = nib.load(path)
    # Reorient to canonical (RAS+) for consistent viewing
    img_canonical = nib.as_closest_canonical(img)
    data = img_canonical.get_fdata()
    return data, img_canonical


def normalize_image(image: np.ndarray, window_center: int = 40, window_width: int = 400) -> np.ndarray:
    """Apply CT windowing for better visualization."""
    min_val = window_center - window_width // 2
    max_val = window_center + window_width // 2
    image = np.clip(image, min_val, max_val)
    image = (image - min_val) / (max_val - min_val)
    return image


def create_mask_colormap(num_classes: int = 9) -> ListedColormap:
    """Create colormap for mask visualization."""
    colors = [HEART_COLORS.get(i, (0.5, 0.5, 0.5, 0.5)) for i in range(num_classes)]
    return ListedColormap(colors)


def get_slice(data: np.ndarray, axis: str, idx: int) -> np.ndarray:
    """Get a 2D slice from 3D volume with proper orientation for viewing."""
    if axis == 'axial':
        # Axial: looking from feet to head, patient supine
        slc = data[:, :, idx]
        slc = np.flipud(slc.T)  # Transpose and flip for radiological view
    elif axis == 'coronal':
        # Coronal: looking from front
        slc = data[:, idx, :]
        slc = np.flipud(slc.T)
    elif axis == 'sagittal':
        # Sagittal: looking from right side
        slc = data[idx, :, :]
        slc = np.flipud(slc.T)
    else:
        raise ValueError(f"Unknown axis: {axis}")
    return slc


def get_num_slices(data: np.ndarray, axis: str) -> int:
    """Get number of slices along an axis."""
    if axis == 'axial':
        return data.shape[2]
    elif axis == 'coronal':
        return data.shape[1]
    elif axis == 'sagittal':
        return data.shape[0]
    else:
        raise ValueError(f"Unknown axis: {axis}")


class HeartViewer:
    """Interactive viewer for heart segmentation."""

    def __init__(self, image: np.ndarray, mask: np.ndarray,
                 window_center: int = 40, window_width: int = 400):
        self.image_raw = image
        self.mask = mask
        self.window_center = window_center
        self.window_width = window_width
        self.image = normalize_image(image, window_center, window_width)
        self.axis = 'axial'
        self.current_slice = get_num_slices(image, self.axis) // 2
        self.alpha = 0.5
        self.show_mask = True

        self.cmap = create_mask_colormap()
        self.setup_figure()

    def setup_figure(self):
        """Set up the matplotlib figure with controls."""
        self.fig = plt.figure(figsize=(14, 10))

        # Main image axis
        self.ax_img = plt.axes([0.1, 0.25, 0.55, 0.7])

        # Slice slider
        ax_slider = plt.axes([0.1, 0.1, 0.55, 0.03])
        max_slice = get_num_slices(self.image, self.axis) - 1
        self.slider = Slider(ax_slider, 'Slice', 0, max_slice,
                            valinit=self.current_slice, valstep=1)
        self.slider.on_changed(self.update_slice)

        # Alpha slider
        ax_alpha = plt.axes([0.1, 0.05, 0.55, 0.03])
        self.alpha_slider = Slider(ax_alpha, 'Opacity', 0, 1, valinit=self.alpha)
        self.alpha_slider.on_changed(self.update_alpha)

        # View selector
        ax_radio = plt.axes([0.7, 0.6, 0.25, 0.15])
        self.radio = RadioButtons(ax_radio, ('Axial', 'Coronal', 'Sagittal'))
        self.radio.on_clicked(self.change_view)

        # Toggle mask button
        ax_toggle = plt.axes([0.7, 0.5, 0.15, 0.05])
        self.btn_toggle = Button(ax_toggle, 'Toggle Mask')
        self.btn_toggle.on_clicked(self.toggle_mask)

        # Window presets
        ax_soft = plt.axes([0.7, 0.4, 0.1, 0.05])
        self.btn_soft = Button(ax_soft, 'Soft Tissue')
        self.btn_soft.on_clicked(lambda x: self.set_window(40, 400))

        ax_lung = plt.axes([0.82, 0.4, 0.1, 0.05])
        self.btn_lung = Button(ax_lung, 'Lung')
        self.btn_lung.on_clicked(lambda x: self.set_window(-600, 1500))

        ax_bone = plt.axes([0.7, 0.33, 0.1, 0.05])
        self.btn_bone = Button(ax_bone, 'Bone')
        self.btn_bone.on_clicked(lambda x: self.set_window(400, 1800))

        ax_cardiac = plt.axes([0.82, 0.33, 0.1, 0.05])
        self.btn_cardiac = Button(ax_cardiac, 'Cardiac')
        self.btn_cardiac.on_clicked(lambda x: self.set_window(50, 350))

        # Legend
        self.draw_legend()

        # Initial draw
        self.update_display()

        # Keyboard navigation
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def draw_legend(self):
        """Draw legend for mask colors."""
        ax_legend = plt.axes([0.7, 0.7, 0.25, 0.25])
        ax_legend.set_title('Heart Structures', fontsize=10)
        ax_legend.axis('off')

        unique_labels = np.unique(self.mask).astype(int)
        y_pos = 0.9
        for label in unique_labels:
            if label == 0:
                continue
            color = HEART_COLORS.get(label, (0.5, 0.5, 0.5, 1))[:3]
            name = HEART_LABELS.get(label, f"Label {label}")
            ax_legend.add_patch(plt.Rectangle((0, y_pos - 0.08), 0.15, 0.08,
                                              facecolor=color, edgecolor='black'))
            ax_legend.text(0.2, y_pos - 0.04, name, fontsize=8, va='center')
            y_pos -= 0.12

    def update_display(self):
        """Update the displayed image."""
        self.ax_img.clear()

        img_slice = get_slice(self.image, self.axis, self.current_slice)
        mask_slice = get_slice(self.mask, self.axis, self.current_slice)

        # Display image with equal aspect ratio to preserve proportions
        self.ax_img.imshow(img_slice, cmap='gray', aspect='equal')

        # Overlay mask
        if self.show_mask:
            masked = np.ma.masked_where(mask_slice == 0, mask_slice)
            self.ax_img.imshow(masked, cmap=self.cmap, alpha=self.alpha,
                              aspect='equal', vmin=0, vmax=8)

        self.ax_img.set_title(f'{self.axis.capitalize()} View - Slice {self.current_slice}')
        self.ax_img.axis('off')
        self.fig.canvas.draw_idle()

    def update_slice(self, val):
        """Update slice index from slider."""
        self.current_slice = int(val)
        self.update_display()

    def update_alpha(self, val):
        """Update mask opacity."""
        self.alpha = val
        self.update_display()

    def change_view(self, label):
        """Change viewing axis."""
        self.axis = label.lower()
        max_slice = get_num_slices(self.image, self.axis) - 1
        self.current_slice = max_slice // 2
        self.slider.valmax = max_slice
        self.slider.set_val(self.current_slice)
        self.update_display()

    def toggle_mask(self, event):
        """Toggle mask visibility."""
        self.show_mask = not self.show_mask
        self.update_display()

    def set_window(self, center: int, width: int):
        """Set CT window level."""
        self.window_center = center
        self.window_width = width
        self.image = normalize_image(self.image_raw, center, width)
        self.update_display()

    def on_scroll(self, event):
        """Handle mouse scroll for slice navigation."""
        if event.inaxes == self.ax_img:
            max_slice = get_num_slices(self.image, self.axis) - 1
            if event.button == 'up':
                self.current_slice = min(self.current_slice + 1, max_slice)
            else:
                self.current_slice = max(self.current_slice - 1, 0)
            self.slider.set_val(self.current_slice)

    def on_key(self, event):
        """Handle keyboard navigation."""
        max_slice = get_num_slices(self.image, self.axis) - 1
        if event.key == 'up' or event.key == 'right':
            self.current_slice = min(self.current_slice + 1, max_slice)
            self.slider.set_val(self.current_slice)
        elif event.key == 'down' or event.key == 'left':
            self.current_slice = max(self.current_slice - 1, 0)
            self.slider.set_val(self.current_slice)
        elif event.key == 'm':
            self.toggle_mask(None)
        elif event.key == 'a':
            self.radio.set_active(0)
        elif event.key == 'c':
            self.radio.set_active(1)
        elif event.key == 's':
            self.radio.set_active(2)

    def show(self):
        """Display the viewer."""
        plt.show()


def save_montage(image: np.ndarray, mask: np.ndarray, output_path: Path,
                 axis: str = 'axial', num_slices: int = 16,
                 window_center: int = 40, window_width: int = 400):
    """Save a montage of slices to an image file."""
    image_norm = normalize_image(image, window_center, window_width)
    cmap = create_mask_colormap()

    total_slices = get_num_slices(image, axis)
    indices = np.linspace(total_slices * 0.2, total_slices * 0.8, num_slices, dtype=int)

    cols = 4
    rows = (num_slices + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        img_slice = get_slice(image_norm, axis, idx)
        mask_slice = get_slice(mask, axis, idx)

        axes[i].imshow(img_slice, cmap='gray', aspect='equal')
        masked = np.ma.masked_where(mask_slice == 0, mask_slice)
        axes[i].imshow(masked, cmap=cmap, alpha=0.5, vmin=0, vmax=8, aspect='equal')
        axes[i].set_title(f'Slice {idx}')
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Heart Segmentation - {axis.capitalize()} View', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Montage saved to {output_path}")


def save_single_slice(image: np.ndarray, mask: np.ndarray, output_path: Path,
                      axis: str = 'axial', slice_idx: Optional[int] = None,
                      window_center: int = 40, window_width: int = 400):
    """Save a single slice visualization."""
    image_norm = normalize_image(image, window_center, window_width)
    cmap = create_mask_colormap()

    if slice_idx is None:
        slice_idx = get_num_slices(image, axis) // 2

    img_slice = get_slice(image_norm, axis, slice_idx)
    mask_slice = get_slice(mask, axis, slice_idx)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img_slice, cmap='gray', aspect='equal')
    axes[0].set_title('CT Image')
    axes[0].axis('off')

    # Mask only
    axes[1].imshow(mask_slice, cmap=cmap, vmin=0, vmax=8, aspect='equal')
    axes[1].set_title('Heart Mask')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(img_slice, cmap='gray', aspect='equal')
    masked = np.ma.masked_where(mask_slice == 0, mask_slice)
    axes[2].imshow(masked, cmap=cmap, alpha=0.5, vmin=0, vmax=8, aspect='equal')
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    # Add legend
    unique_labels = np.unique(mask_slice).astype(int)
    legend_elements = []
    for label in unique_labels:
        if label == 0:
            continue
        color = HEART_COLORS.get(label, (0.5, 0.5, 0.5, 1))[:3]
        name = HEART_LABELS.get(label, f"Label {label}")
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color,
                                             edgecolor='black', label=name))

    if legend_elements:
        fig.legend(handles=legend_elements, loc='lower center', ncol=min(4, len(legend_elements)),
                   bbox_to_anchor=(0.5, 0.02))

    plt.suptitle(f'{axis.capitalize()} View - Slice {slice_idx}', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Image saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize heart segmentation masks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive viewer
  python viz.py --image ct_scan.nii.gz --mask heart_mask.nii.gz

  # Save montage
  python viz.py --image ct_scan.nii.gz --mask heart_mask.nii.gz --save montage.png --montage

  # Save specific slice
  python viz.py --image ct_scan.nii.gz --mask heart_mask.nii.gz --save slice.png --slice 100

Keyboard shortcuts (interactive mode):
  Arrow keys / Scroll: Navigate slices
  m: Toggle mask visibility
  a/c/s: Switch to Axial/Coronal/Sagittal view
        """
    )

    parser.add_argument('--image', '-i', type=Path, required=True,
                        help='Input CT image (NIfTI format)')
    parser.add_argument('--mask', '-m', type=Path, required=True,
                        help='Segmentation mask (NIfTI format)')
    parser.add_argument('--save', '-s', type=Path, default=None,
                        help='Save visualization to file instead of interactive display')
    parser.add_argument('--axis', '-a', choices=['axial', 'coronal', 'sagittal'],
                        default='axial', help='Viewing axis (default: axial)')
    parser.add_argument('--slice', type=int, default=None,
                        help='Specific slice index to display')
    parser.add_argument('--montage', action='store_true',
                        help='Save as montage of multiple slices')
    parser.add_argument('--num-slices', type=int, default=16,
                        help='Number of slices in montage (default: 16)')
    parser.add_argument('--window-center', '-wc', type=int, default=40,
                        help='CT window center (default: 40 for soft tissue)')
    parser.add_argument('--window-width', '-ww', type=int, default=400,
                        help='CT window width (default: 400 for soft tissue)')

    args = parser.parse_args()

    # Validate inputs
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.mask.exists():
        raise FileNotFoundError(f"Mask not found: {args.mask}")

    # Load data
    print(f"Loading image: {args.image}")
    image, _ = load_nifti(args.image)
    print(f"Loading mask: {args.mask}")
    mask, _ = load_nifti(args.mask)

    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Unique mask values: {np.unique(mask).astype(int)}")

    # Check shape compatibility
    if image.shape != mask.shape:
        print(f"Warning: Shape mismatch! Image: {image.shape}, Mask: {mask.shape}")

    if args.save:
        if args.montage:
            save_montage(image, mask, args.save, args.axis, args.num_slices,
                        args.window_center, args.window_width)
        else:
            save_single_slice(image, mask, args.save, args.axis, args.slice,
                             args.window_center, args.window_width)
    else:
        viewer = HeartViewer(image, mask, args.window_center, args.window_width)
        viewer.show()


if __name__ == '__main__':
    main()
