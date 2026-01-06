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
import SimpleITK as sitk
import torch
from monai.transforms import SaveImage
from monai.transforms.utils import allow_missing_keys_mode

from salt.input_pipeline import (
    IntensityProperties,
    get_validation_transforms,
    get_postprocess_transforms,
)
from salt.core.label_tree import LabelTree
from salt.utils.inference import sliding_window_inference_with_reduction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Path to tree-labels file (defines the hierarchical structure)
TREE_LABELS_FILE = Path(__file__).parent / "labels" / "saros+totalseg" / "tree-labels.txt"

# Heart structure names to search for
HEART_LABEL_NAMES = [
    "heart_myocardium",
    "heart_atrium_left",
    "heart_ventricle_left",
    "heart_atrium_right",
    "heart_ventricle_right",
]


def load_tree_labels(tree_labels_file: Path = TREE_LABELS_FILE) -> List[Tuple[str, ...]]:
    """Load tree labels from tree-labels.txt file.

    Returns list of label tuples, e.g., [('body', 'heart', 'myocardium'), ...]
    """
    if not tree_labels_file.exists():
        logger.warning(f"Tree labels file not found: {tree_labels_file}")
        return []

    labels = []
    with open(tree_labels_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and line != "-":
                # Split by comma to get path components
                parts = tuple(line.split(","))
                labels.append(parts)
            else:
                labels.append(None)  # Placeholder for skipped lines

    logger.info(f"Loaded {len([l for l in labels if l])} labels from {tree_labels_file}")
    return labels


def build_label_tree(tree_labels: List[Tuple[str, ...]]) -> LabelTree:
    """Build a LabelTree from tree labels (same as during training)."""
    tree = LabelTree()

    for label in tree_labels:
        if label is not None:
            tree.add(*label)

    # Sort the tree (same as during training)
    tree.optimize()

    return tree


def find_heart_indices(tree: LabelTree) -> Dict[str, int]:
    """Find indices of heart labels in the label tree vocabulary.

    Returns dict mapping label name to vocabulary index (0-indexed, excluding root).
    """
    heart_indices = {}
    vocabulary = tree.labels  # List of label tuples (excluding root)

    for idx, label_tuple in enumerate(vocabulary):
        # Get the last element (most specific label name)
        label_name = label_tuple[-1] if label_tuple else ""

        if label_name in HEART_LABEL_NAMES:
            heart_indices[label_name] = idx
            logger.info(f"Found heart label: {label_name} at index {idx}")

    return heart_indices


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


def get_leaf_node_info(adjacency_matrix: np.ndarray) -> Tuple[np.ndarray, Dict[int, int], Dict[int, int]]:
    """Get leaf node indices and mappings.

    Returns:
        leaf_nodes: Array of original indices that are leaf nodes
        pruned_to_original: Dict mapping pruned leaf index to original class index
        original_to_pruned: Dict mapping original class index to pruned leaf index
    """
    leaf_nodes = np.where(adjacency_matrix[1:, 1:].sum(axis=1) == 0)[0]
    pruned_to_original = {i: int(leaf_nodes[i]) for i in range(len(leaf_nodes))}
    original_to_pruned = {int(leaf_nodes[i]): i for i in range(len(leaf_nodes))}

    return leaf_nodes, pruned_to_original, original_to_pruned


def show_all_labels(config: dict) -> None:
    """Print all labels in the model."""
    adjacency_matrix = config["adjacency_matrix"]
    leaf_nodes, pruned_to_original, original_to_pruned = get_leaf_node_info(adjacency_matrix)

    # Build label tree to get vocabulary
    tree_labels = load_tree_labels()
    tree = build_label_tree(tree_labels)
    vocabulary = tree.labels

    # Find heart indices
    heart_indices = find_heart_indices(tree)

    print("\n" + "=" * 80)
    print("SALT MODEL LABELS (saros+totalseg)")
    print("=" * 80)
    print(f"\nTotal vocabulary size: {len(vocabulary)}")
    print(f"Adjacency matrix shape: {adjacency_matrix.shape}")
    print(f"Number of leaf nodes (segmentable classes): {len(leaf_nodes)}")

    print("\n--- LEAF NODES (output classes) ---")
    print(f"{'Pruned':>8} {'Original':>10} {'Label Path':<55} {'Heart?'}")
    print("-" * 80)

    for pruned_idx in range(len(leaf_nodes)):
        original_idx = pruned_to_original[pruned_idx]
        if original_idx < len(vocabulary):
            label_tuple = vocabulary[original_idx]
            label_path = " > ".join(label_tuple)
            label_name = label_tuple[-1] if label_tuple else ""
        else:
            label_path = f"unknown_{original_idx}"
            label_name = ""

        is_heart = label_name in HEART_LABEL_NAMES
        marker = "<-- HEART" if is_heart else ""

        # Truncate long paths
        if len(label_path) > 55:
            label_path = "..." + label_path[-52:]

        print(f"{pruned_idx:>8} {original_idx:>10} {label_path:<55} {marker}")

    print("\n--- HEART LABELS SUMMARY ---")
    for name in HEART_LABEL_NAMES:
        if name in heart_indices:
            orig_idx = heart_indices[name]
            if orig_idx in original_to_pruned:
                pruned_idx = original_to_pruned[orig_idx]
                print(f"  {name}: original={orig_idx}, pruned={pruned_idx}")
            else:
                print(f"  {name}: original={orig_idx}, NOT A LEAF NODE!")
        else:
            print(f"  {name}: NOT FOUND IN VOCABULARY!")

    print("=" * 80 + "\n")


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


def extract_heart_from_prediction(
    prediction: np.ndarray,
    heart_indices: Dict[str, int],
    original_to_pruned: Dict[int, int],
    binary: bool = False,
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Extract heart structures from full segmentation prediction.

    Args:
        prediction: Full segmentation output with pruned leaf indices
        heart_indices: Dict mapping heart label name to original index
        original_to_pruned: Mapping from original class index to pruned leaf index
        binary: If True, output binary mask

    Returns:
        heart_mask: Mask with only heart structures
        label_mapping: Dict of output value -> label name
    """
    heart_mask = np.zeros_like(prediction, dtype=np.uint8)
    label_mapping = {}

    output_value = 1
    for label_name in HEART_LABEL_NAMES:
        if label_name not in heart_indices:
            logger.warning(f"  {label_name}: not found in vocabulary, skipping")
            continue

        original_idx = heart_indices[label_name]

        if original_idx not in original_to_pruned:
            logger.warning(f"  {label_name} (idx={original_idx}) is not a leaf node, skipping")
            continue

        pruned_idx = original_to_pruned[original_idx]
        mask_locations = prediction == pruned_idx
        voxel_count = np.sum(mask_locations)

        if voxel_count > 0:
            if binary:
                heart_mask[mask_locations] = 1
            else:
                heart_mask[mask_locations] = output_value
                label_mapping[output_value] = label_name
                output_value += 1

            logger.info(f"  {label_name}: {voxel_count:,} voxels (original={original_idx}, pruned={pruned_idx})")
        else:
            logger.info(f"  {label_name}: 0 voxels (original={original_idx}, pruned={pruned_idx})")

    if binary:
        label_mapping = {1: "heart"}

    total_heart_voxels = np.sum(heart_mask > 0)
    logger.info(f"  Total heart voxels: {total_heart_voxels:,}")

    return heart_mask, label_mapping


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

    logger.info(f"Config keys: {list(config.keys())}")

    # Build label tree from tree-labels.txt (same as during training)
    logger.info("Building label tree...")
    tree_labels = load_tree_labels()
    tree = build_label_tree(tree_labels)

    # Find heart label indices in the vocabulary
    heart_indices = find_heart_indices(tree)

    if not heart_indices:
        raise ValueError("Could not find any heart labels in vocabulary!")

    # Get leaf node mappings
    adjacency_matrix = config["adjacency_matrix"]
    leaf_nodes, pruned_to_original, original_to_pruned = get_leaf_node_info(adjacency_matrix)
    logger.info(f"Number of leaf nodes: {len(leaf_nodes)}")

    # Verify heart labels are leaf nodes
    logger.info("Heart label indices:")
    for name in HEART_LABEL_NAMES:
        if name in heart_indices:
            orig_idx = heart_indices[name]
            if orig_idx in original_to_pruned:
                logger.info(f"  {name}: original={orig_idx} -> pruned={original_to_pruned[orig_idx]}")
            else:
                logger.warning(f"  {name}: original={orig_idx} -> NOT A LEAF NODE!")
        else:
            logger.warning(f"  {name}: NOT FOUND IN VOCABULARY!")

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
        logger.info(f"Loading and preprocessing image from {nifti_path}...")
        with allow_missing_keys_mode(pre_processing):
            example = pre_processing({"image": nifti_path})

        logger.info(f"Preprocessed image shape: {example['image'].shape}")

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
                .to(torch.int64)
            )

        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f} seconds")

        # Apply post-processing (same as predict.py - includes KeepLargestConnectedComponent)
        logger.info("Applying post-processing...")
        postprocess = get_postprocess_transforms(output_dir=None)
        processed = postprocess({
            "pred": prediction,
            "image_meta_dict": example["image_meta_dict"]
        })
        prediction_np = processed["pred"].numpy()[0]  # Remove channel dim

        logger.info(f"Post-processed prediction shape: {prediction_np.shape}")

        # Debug: show prediction stats
        unique_values = np.unique(prediction_np)
        logger.info(f"Unique values in prediction: {len(unique_values)} (range: {unique_values.min()} - {unique_values.max()})")

        # Extract heart mask
        logger.info("Extracting heart structures...")
        heart_mask, label_mapping = extract_heart_from_prediction(
            prediction_np,
            heart_indices,
            original_to_pruned,
            binary=binary,
        )

        # Save using the metadata from preprocessing
        input_name = input_path.name if input_path.is_dir() else input_path.stem.replace('.nii', '')
        if input_name.endswith("_temp"):
            input_name = input_name[:-5]
        output_path = output_dir / f"{input_name}_heart_mask.nii.gz"

        # Create NIfTI with proper metadata
        saver = SaveImage(
            output_dir=str(output_dir),
            output_postfix="",
            output_ext=".nii.gz",
            output_dtype=np.uint8,
            separate_folder=False,
        )

        # Add channel dimension back for SaveImage
        heart_mask_tensor = torch.from_numpy(heart_mask).unsqueeze(0)

        # Update metadata to use our filename
        meta_dict = dict(example["image_meta_dict"])
        meta_dict["filename_or_obj"] = str(output_path)

        saver(heart_mask_tensor, meta_dict)

        logger.info(f"Heart mask saved to {output_path}")

        # Save label mapping
        label_file = output_dir / f"{input_name}_heart_labels.txt"
        with open(label_file, "w") as f:
            f.write("# Heart Segmentation Labels\n")
            f.write("# Value: Label Name\n")
            for value, name in sorted(label_mapping.items()):
                count = np.sum(heart_mask == value)
                f.write(f"{value}: {name} ({count:,} voxels)\n")
        logger.info(f"Labels saved to {label_file}")

        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("HEART SEGMENTATION COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Time: {inference_time:.2f}s")
        logger.info(f"Total heart voxels: {np.sum(heart_mask > 0):,}")
        for value, name in sorted(label_mapping.items()):
            logger.info(f"  {value}: {name} ({np.sum(heart_mask == value):,} voxels)")
        logger.info("=" * 50)

        return output_path

    finally:
        if is_temp and not keep_temp:
            nifti_path.unlink(missing_ok=True)
            logger.info("Cleaned up temp files")


def main():
    parser = ArgumentParser(
        description="Heart Segmentation - extracts heart structures from CT using SALT",
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
