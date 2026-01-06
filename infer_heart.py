#!/usr/bin/env python3
"""
Heart Segmentation Inference Script for SALT

This script performs heart segmentation from CT images (DICOM folder or NIfTI format).
Runs full SALT inference, then extracts only heart structures.

Usage:
    # From DICOM folder:
    python infer_heart.py --input /path/to/dicom_folder --output /path/to/output

    # From NIfTI file:
    python infer_heart.py --input /path/to/image.nii.gz --output /path/to/output

    # Binary mask (all heart structures combined as 1):
    python infer_heart.py --input /path/to/dicom_folder --output ./output --binary

    # Show available labels:
    python infer_heart.py --show-labels

Author: Generated based on SALT framework
"""

import logging
import pickle
import time
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
from monai.transforms.utils import allow_missing_keys_mode

from salt.input_pipeline import (
    IntensityProperties,
    get_validation_transforms,
    get_postprocess_transforms,
)
from salt.utils.inference import sliding_window_inference_with_reduction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Heart structure keywords to search for in vocabulary
HEART_KEYWORDS = [
    "heart_myocardium",
    "heart_atrium_left",
    "heart_ventricle_left",
    "heart_atrium_right",
    "heart_ventricle_right",
    "myocardium",
    "atrium_left",
    "ventricle_left",
    "atrium_right",
    "ventricle_right",
]


def argmax_leaves(
    inputs: torch.Tensor,
    adjacency_matrix: np.ndarray,
    dim: int = 1,
    pruned: bool = True,
) -> torch.Tensor:
    """Compute argmax over leaf nodes in the label tree.

    This is copied from predict.py to ensure identical behavior.
    """
    leave_nodes = np.where(adjacency_matrix[1:, 1:].sum(axis=1) == 0)[0]
    indices = np.arange(adjacency_matrix.shape[0] - 1, dtype=np.int32)
    indices = indices[leave_nodes]
    y_pred_leaves = inputs[:, leave_nodes]
    y_pred_leave_idx = torch.argmax(y_pred_leaves, axis=dim)
    if pruned:
        return y_pred_leave_idx
    return torch.tensor(indices).to(inputs.device)[y_pred_leave_idx]


def get_leaf_node_info(adjacency_matrix: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    """Get leaf node indices and mapping from pruned index to original index.

    Returns:
        leaf_nodes: Array of original indices that are leaf nodes
        pruned_to_original: Dict mapping pruned leaf index to original class index
    """
    # Find nodes with no children (sum of outgoing edges = 0)
    leaf_nodes = np.where(adjacency_matrix[1:, 1:].sum(axis=1) == 0)[0]

    # Create mapping: pruned leaf index -> original class index
    pruned_to_original = {i: int(leaf_nodes[i]) for i in range(len(leaf_nodes))}

    return leaf_nodes, pruned_to_original


def find_heart_labels_in_vocabulary(vocabulary: List[Tuple[str, ...]]) -> Dict[str, int]:
    """Find heart-related labels in the vocabulary.

    Args:
        vocabulary: List of label tuples, e.g., [('body', 'heart', 'myocardium'), ...]

    Returns:
        Dict mapping label name to its index in vocabulary (0-indexed, excluding root)
    """
    heart_labels = {}

    for idx, label_tuple in enumerate(vocabulary):
        # Get the last element (most specific label name)
        label_name = label_tuple[-1] if label_tuple else ""
        full_path = ",".join(label_tuple)

        # Check if this is a heart-related label
        for keyword in HEART_KEYWORDS:
            if keyword in label_name.lower() or keyword in full_path.lower():
                # Use a clean name
                clean_name = label_name.replace("_", " ").title()
                heart_labels[label_name] = idx
                logger.debug(f"Found heart label: {label_name} at index {idx}")
                break

    return heart_labels


def show_all_labels(config: dict) -> None:
    """Print all labels in the model vocabulary."""
    adjacency_matrix = config["adjacency_matrix"]
    vocabulary = config.get("vocabulary", [])

    leaf_nodes, _ = get_leaf_node_info(adjacency_matrix)

    print("\n" + "=" * 60)
    print("SALT MODEL LABELS")
    print("=" * 60)

    if vocabulary:
        print(f"\nTotal vocabulary size: {len(vocabulary)}")
        print(f"Number of leaf nodes (segmentable classes): {len(leaf_nodes)}")

        print("\n--- LEAF NODES (output classes) ---")
        for pruned_idx, original_idx in enumerate(leaf_nodes):
            if original_idx < len(vocabulary):
                label = vocabulary[original_idx]
                label_str = " > ".join(label) if isinstance(label, tuple) else str(label)

                # Highlight heart-related labels
                is_heart = any(kw in label_str.lower() for kw in ["heart", "myocard", "atrium", "ventricle"])
                marker = " <-- HEART" if is_heart else ""

                print(f"  [{pruned_idx:3d}] -> original[{original_idx:3d}]: {label_str}{marker}")
    else:
        print("\nNo vocabulary found in config. Adjacency matrix shape:", adjacency_matrix.shape)
        print(f"Number of leaf nodes: {len(leaf_nodes)}")
        print(f"Leaf node original indices: {leaf_nodes.tolist()}")

    print("=" * 60 + "\n")


def dicom_to_nifti(dicom_path: Path, output_path: Optional[Path] = None) -> Path:
    """Convert DICOM folder to NIfTI format."""
    logger.info(f"Converting DICOM from {dicom_path} to NIfTI...")

    reader = sitk.ImageSeriesReader()
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_path))

    if len(series_ids) > 0:
        logger.info(f"Found {len(series_ids)} DICOM series")
        dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            str(dicom_path), series_ids[0]
        )
        logger.info(f"Using series with {len(dicom_names)} slices")
    else:
        dcm_files = list(dicom_path.glob("*.dcm")) + list(dicom_path.glob("*.DCM"))

        if len(dcm_files) == 0:
            all_files = sorted([f for f in dicom_path.iterdir() if f.is_file()])
            if len(all_files) == 0:
                raise ValueError(f"No files found in {dicom_path}")

            logger.info(f"No .dcm files found, trying {len(all_files)} files without extension...")
            dicom_names = [str(f) for f in all_files]
        else:
            dcm_files = sorted(dcm_files, key=lambda x: x.name)
            dicom_names = [str(f) for f in dcm_files]
            logger.info(f"Found {len(dicom_names)} .dcm files")

    if len(dicom_names) == 0:
        raise ValueError(f"No DICOM files found in {dicom_path}")

    reader.SetFileNames(dicom_names)
    reader.SetImageIO("GDCMImageIO")
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    logger.info(f"Reading {len(dicom_names)} DICOM slices...")
    image = reader.Execute()

    logger.info(f"Image size: {image.GetSize()}, spacing: {image.GetSpacing()}")

    if output_path is None:
        output_path = dicom_path.parent / f"{dicom_path.name}_converted.nii.gz"

    sitk.WriteImage(image, str(output_path))
    logger.info(f"Saved to {output_path}")

    return output_path


def load_input(input_path: Path, temp_dir: Optional[Path] = None) -> Tuple[Path, bool]:
    """Load input, converting from DICOM if it's a folder."""
    if input_path.is_dir():
        logger.info(f"Input is DICOM folder...")
        if temp_dir is None:
            temp_dir = input_path.parent
        nifti_path = dicom_to_nifti(
            input_path,
            temp_dir / f"{input_path.name}_temp.nii.gz"
        )
        return nifti_path, True
    elif input_path.suffix.lower() in [".gz", ".nii"]:
        logger.info(f"Input is NIfTI file")
        return input_path, False
    elif input_path.suffix.lower() == ".dcm":
        logger.info(f"Input is single DICOM, using parent folder...")
        if temp_dir is None:
            temp_dir = input_path.parent
        nifti_path = dicom_to_nifti(
            input_path.parent,
            temp_dir / f"{input_path.parent.name}_temp.nii.gz"
        )
        return nifti_path, True
    else:
        raise ValueError(
            f"Unsupported format: {input_path}\n"
            "Supported: DICOM folder, .nii, .nii.gz, .dcm"
        )


def extract_heart_mask(
    prediction: np.ndarray,
    config: dict,
    binary: bool = False,
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Extract heart structures from full body segmentation.

    Args:
        prediction: Full segmentation output with pruned leaf indices
        config: Model config containing adjacency_matrix and vocabulary
        binary: If True, output binary mask

    Returns:
        heart_mask: Mask with only heart structures
        label_mapping: Dict of output value -> label name
    """
    adjacency_matrix = config["adjacency_matrix"]
    vocabulary = config.get("vocabulary", [])

    leaf_nodes, pruned_to_original = get_leaf_node_info(adjacency_matrix)

    # Create reverse mapping: original index -> pruned leaf index
    original_to_pruned = {v: k for k, v in pruned_to_original.items()}

    # Find heart labels in vocabulary
    heart_original_indices = {}

    if vocabulary:
        for original_idx, label_tuple in enumerate(vocabulary):
            label_name = label_tuple[-1] if label_tuple else ""
            full_path = ",".join(label_tuple).lower()

            # Check for heart-related keywords
            if any(kw in full_path for kw in ["heart", "myocard", "atrium", "ventricle"]):
                # Only include if it's a leaf node (actually segmented)
                if original_idx in original_to_pruned:
                    heart_original_indices[label_name] = original_idx
                    logger.info(f"  Heart label found: {label_name} (original={original_idx}, pruned={original_to_pruned[original_idx]})")
    else:
        # Fallback: use known SAROS+TotalSeg indices if no vocabulary
        logger.warning("No vocabulary in config, using fallback indices")
        fallback_indices = {
            "heart_myocardium": 57,
            "heart_atrium_left": 58,
            "heart_ventricle_left": 59,
            "heart_atrium_right": 60,
            "heart_ventricle_right": 61,
        }
        for name, idx in fallback_indices.items():
            if idx in original_to_pruned:
                heart_original_indices[name] = idx

    if not heart_original_indices:
        logger.warning("No heart labels found! Dumping leaf node info for debugging...")
        for pruned_idx in range(min(10, len(leaf_nodes))):
            logger.warning(f"  Pruned {pruned_idx} -> Original {leaf_nodes[pruned_idx]}")
        raise ValueError("Could not find heart labels in model vocabulary")

    # Extract heart mask
    heart_mask = np.zeros_like(prediction, dtype=np.uint8)
    label_mapping = {}

    output_value = 1
    for label_name, original_idx in sorted(heart_original_indices.items()):
        pruned_idx = original_to_pruned[original_idx]
        mask_locations = prediction == pruned_idx

        if np.any(mask_locations):
            if binary:
                heart_mask[mask_locations] = 1
            else:
                heart_mask[mask_locations] = output_value
                label_mapping[output_value] = label_name
                output_value += 1

            voxel_count = np.sum(mask_locations)
            logger.info(f"  {label_name}: {voxel_count} voxels (pruned_idx={pruned_idx})")

    if binary:
        label_mapping = {1: "heart"}

    total_heart_voxels = np.sum(heart_mask > 0)
    logger.info(f"  Total heart voxels: {total_heart_voxels}")

    return heart_mask, label_mapping


def save_mask(
    mask: np.ndarray,
    reference_image: nib.Nifti1Image,
    output_path: Path,
    label_mapping: Dict[int, str],
) -> None:
    """Save segmentation mask as NIfTI file."""
    mask_nifti = nib.Nifti1Image(
        mask.astype(np.uint8),
        affine=reference_image.affine,
        header=reference_image.header.copy()
    )
    mask_nifti.header.set_data_dtype(np.uint8)

    nib.save(mask_nifti, output_path)
    logger.info(f"Heart mask saved to {output_path}")

    # Save label mapping
    label_file = output_path.parent / f"{output_path.stem.replace('.nii', '')}_labels.txt"
    with open(label_file, "w") as f:
        f.write("# Heart Segmentation Labels\n")
        for value, name in sorted(label_mapping.items()):
            f.write(f"{value}: {name}\n")
    logger.info(f"Labels saved to {label_file}")


def run_inference(
    input_path: Path,
    output_dir: Path,
    model_file: Path,
    config_file: Path,
    binary: bool = False,
    keep_temp: bool = False,
) -> Path:
    """Run heart segmentation inference."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    logger.info("Loading config...")
    with config_file.open("rb") as f:
        config = pickle.load(f)

    # Debug: show what's in config
    logger.info(f"Config keys: {list(config.keys())}")
    if "vocabulary" in config:
        logger.info(f"Vocabulary size: {len(config['vocabulary'])}")

    # Load model
    logger.info("Loading model...")
    model = torch.jit.load(model_file)
    model.cuda()
    model.eval()
    torch._C._jit_set_profiling_executor(False)

    pre_processing = get_validation_transforms(
        spacing=config["model"]["voxel_spacing"],
        info=None,
        intensity_properties=(
            IntensityProperties(
                mean=config["intensity_properties"]["mean"],
                std=config["intensity_properties"]["std"],
            )
            if config["intensity_properties"] is not None
            else None
        ),
    )

    # Load input
    nifti_path, is_temp = load_input(input_path, output_dir)

    try:
        logger.info(f"Loading image from {nifti_path}...")
        with allow_missing_keys_mode(pre_processing):
            example = pre_processing({"image": nifti_path})

        reference_image = nib.load(nifti_path)

        # Run inference (same as predict.py)
        logger.info("Running inference...")
        start_time = time.time()

        with torch.cuda.amp.autocast(), torch.no_grad():
            prediction = (
                sliding_window_inference_with_reduction(
                    inputs=example["image"].unsqueeze(0).cuda(),
                    roi_size=config["model"]["roi_size"],
                    sw_batch_size=2,
                    predictor=model,
                    progress=True,
                    overlap=0.5,
                    mode="gaussian",
                    cval=(
                        (-1024 - config["intensity_properties"]["mean"])
                        / config["intensity_properties"]["std"]
                        if config["intensity_properties"] is not None
                        else 0.0
                    ),
                    reduction_fn=partial(
                        argmax_leaves,
                        adjacency_matrix=config["adjacency_matrix"]
                    ),
                )
                .cpu()
                .numpy()
            )

        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f} seconds")

        # Debug: show prediction stats
        prediction = prediction[0]  # Remove batch dimension
        unique_values = np.unique(prediction)
        logger.info(f"Prediction shape: {prediction.shape}")
        logger.info(f"Unique values in prediction: {len(unique_values)} (range: {unique_values.min()} - {unique_values.max()})")

        # Extract heart mask
        logger.info("Extracting heart structures...")
        heart_mask, label_mapping = extract_heart_mask(
            prediction,
            config,
            binary=binary,
        )

        # Save
        input_name = input_path.name if input_path.is_dir() else input_path.stem
        if input_name.endswith("_temp"):
            input_name = input_name[:-5]
        output_path = output_dir / f"{input_name}_heart_mask.nii.gz"

        save_mask(heart_mask, reference_image, output_path, label_mapping)

        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("HEART SEGMENTATION COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Time: {inference_time:.2f}s")
        logger.info(f"Total voxels: {np.sum(heart_mask > 0)}")
        for value, name in sorted(label_mapping.items()):
            logger.info(f"  {value}: {name} ({np.sum(heart_mask == value)} voxels)")
        logger.info("=" * 50)

        return output_path

    finally:
        if is_temp and not keep_temp:
            nifti_path.unlink(missing_ok=True)
            logger.info("Cleaned up temp files")


def main():
    parser = ArgumentParser(
        description="Heart Segmentation - segments heart organ from CT",
        epilog="""
Examples:
  # DICOM folder:
  python infer_heart.py --input /path/to/dicom_folder --output ./output

  # NIfTI file:
  python infer_heart.py --input image.nii.gz --output ./output

  # Binary mask:
  python infer_heart.py --input /path/to/dicom_folder --output ./output --binary

  # Show all labels in model:
  python infer_heart.py --show-labels
        """
    )

    parser.add_argument(
        "--input", "-i", type=Path,
        help="Input DICOM folder or NIfTI file"
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        help="Output directory"
    )
    parser.add_argument(
        "--model-file", type=Path,
        default=Path("models/foobar-31/model.pt"),
        help="Path to model file"
    )
    parser.add_argument(
        "--config-file", type=Path,
        default=Path("models/foobar-31/config.pkl"),
        help="Path to config file"
    )
    parser.add_argument(
        "--binary", action="store_true",
        help="Output binary mask (all heart as value 1)"
    )
    parser.add_argument(
        "--keep-temp", action="store_true",
        help="Keep temporary files"
    )
    parser.add_argument(
        "--show-labels", action="store_true",
        help="Show all labels in the model and exit"
    )

    args = parser.parse_args()

    if not args.config_file.exists():
        raise FileNotFoundError(f"Config not found: {args.config_file}")

    # Show labels mode
    if args.show_labels:
        with args.config_file.open("rb") as f:
            config = pickle.load(f)
        show_all_labels(config)
        return

    # Normal inference mode
    if args.input is None:
        parser.error("--input is required for inference")
    if args.output is None:
        parser.error("--output is required for inference")

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")
    if not args.model_file.exists():
        raise FileNotFoundError(f"Model not found: {args.model_file}")

    output_path = run_inference(
        input_path=args.input,
        output_dir=args.output,
        model_file=args.model_file,
        config_file=args.config_file,
        binary=args.binary,
        keep_temp=args.keep_temp,
    )

    print(f"\nHeart mask saved to: {output_path}")


if __name__ == "__main__":
    main()
