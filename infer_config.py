#!/usr/bin/env python3
"""
SALT Segmentation Inference with Config File

Reads a YAML config file to determine which structures to extract and how to group them.

Usage:
    python infer_config.py --config configs/heart_binary.yaml --input image.nii.gz --output ./output
"""

import logging
import pickle
import time
import re
import yaml
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
from monai.transforms.utils import allow_missing_keys_mode
from scipy.ndimage import zoom, label, generate_binary_structure

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

# Path to tree-labels file
TREE_LABELS_FILE = Path(__file__).parent / "labels" / "saros+totalseg" / "tree-labels.txt"


def load_config(config_path: Path) -> dict:
    """Load and parse YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_groups_from_config(config: dict) -> Dict[str, List[str]]:
    """
    Parse group definitions from config.

    Looks for patterns like:
        SkeletonName: 'skeleton'
        SkeletonList: ['bone1', 'bone2', ...]

    Returns dict: {'skeleton': ['bone1', 'bone2', ...]}
    """
    groups = {}

    # Find all *Name keys
    name_keys = [k for k in config.keys() if k.endswith('Name') and k != 'TargetClass']

    for name_key in name_keys:
        # Get the prefix (e.g., "Skeleton" from "SkeletonName")
        prefix = name_key[:-4]  # Remove "Name"
        list_key = f"{prefix}List"

        if list_key in config:
            group_name = config[name_key]
            group_members = config[list_key]
            groups[group_name] = group_members
            logger.info(f"Found group '{group_name}' with {len(group_members)} members")

    return groups


def load_tree_labels(tree_labels_file: Path = TREE_LABELS_FILE) -> List[Tuple[str, ...]]:
    """Load tree labels from tree-labels.txt file."""
    labels = []
    with open(tree_labels_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and line != "-":
                parts = tuple(line.split(","))
                labels.append(parts)
            else:
                labels.append(None)
    return labels


def build_label_tree(tree_labels: List[Tuple[str, ...]]) -> LabelTree:
    """Build a LabelTree from tree labels."""
    tree = LabelTree()
    for label in tree_labels:
        if label is not None:
            tree.add(*label)
    tree.optimize()
    return tree


def get_label_indices(tree: LabelTree, label_names: List[str]) -> Dict[str, int]:
    """Find indices for given label names in the vocabulary."""
    indices = {}
    vocabulary = tree.labels

    for idx, label_tuple in enumerate(vocabulary):
        label_name = label_tuple[-1] if label_tuple else ""
        if label_name in label_names:
            indices[label_name] = idx

    return indices


def argmax_leaves(
    inputs: torch.Tensor,
    adjacency_matrix: np.ndarray,
    dim: int = 1,
    pruned: bool = True,
) -> torch.Tensor:
    """Compute argmax over leaf nodes in the label tree."""
    leave_nodes = np.where(adjacency_matrix[1:, 1:].sum(axis=1) == 0)[0]
    indices = np.arange(adjacency_matrix.shape[0] - 1, dtype=np.int32)
    indices = indices[leave_nodes]
    y_pred_leaves = inputs[:, leave_nodes]
    y_pred_leave_idx = torch.argmax(y_pred_leaves, axis=dim)
    if pruned:
        return y_pred_leave_idx
    return torch.tensor(indices).to(inputs.device)[y_pred_leave_idx]


def get_leaf_node_info(adjacency_matrix: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    """Get leaf node indices and mapping from original to pruned."""
    leaf_nodes = np.where(adjacency_matrix[1:, 1:].sum(axis=1) == 0)[0]
    original_to_pruned = {int(leaf_nodes[i]): i for i in range(len(leaf_nodes))}
    return leaf_nodes, original_to_pruned


def remove_small_objects(mask: np.ndarray, keep_n: int = 1) -> np.ndarray:
    """Remove small disconnected objects, keeping only N largest."""
    if np.sum(mask > 0) == 0:
        return mask

    structure = generate_binary_structure(3, 3)
    labeled, num_components = label(mask > 0, structure=structure)

    if num_components <= keep_n:
        return mask

    # Get component sizes
    sizes = []
    for i in range(1, num_components + 1):
        sizes.append((i, np.sum(labeled == i)))
    sizes.sort(key=lambda x: x[1], reverse=True)

    # Keep largest N
    keep_ids = set(comp_id for comp_id, _ in sizes[:keep_n])
    keep_mask = np.isin(labeled, list(keep_ids))

    return mask * keep_mask


def dicom_to_nifti(dicom_path: Path, output_path: Optional[Path] = None) -> Path:
    """Convert DICOM folder to NIfTI format."""
    logger.info(f"Converting DICOM from {dicom_path} to NIfTI...")

    reader = sitk.ImageSeriesReader()
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_path))

    if len(series_ids) > 0:
        dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_path), series_ids[0])
    else:
        all_files = sorted([f for f in dicom_path.iterdir() if f.is_file()])
        dicom_names = [str(f) for f in all_files]

    reader.SetFileNames(dicom_names)
    reader.SetImageIO("GDCMImageIO")
    image = reader.Execute()

    if output_path is None:
        output_path = dicom_path.parent / f"{dicom_path.name}_converted.nii.gz"

    sitk.WriteImage(image, str(output_path))
    return output_path


def load_input(input_path: Path, temp_dir: Optional[Path] = None) -> Tuple[Path, bool]:
    """Load input, converting from DICOM if needed."""
    if input_path.is_dir():
        if temp_dir is None:
            temp_dir = input_path.parent
        nifti_path = dicom_to_nifti(input_path, temp_dir / f"{input_path.name}_temp.nii.gz")
        return nifti_path, True
    else:
        return input_path, False


def run_inference(
    config_path: Path,
    input_path: Path,
    output_dir: Path,
    model_file: Path,
    config_file: Path,
    keep_temp: bool = False,
) -> Path:
    """Run segmentation inference using config file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load user config
    logger.info(f"Loading config: {config_path}")
    user_config = load_config(config_path)

    logger.info(f"Target: {user_config.get('Target', 'Unknown')}")
    logger.info(f"TargetClass: {user_config.get('TargetClass', [])}")

    # Parse group definitions
    groups = parse_groups_from_config(user_config)
    target_classes = user_config.get('TargetClass', [])
    flip_xyz = user_config.get('FlipXYZ', [False, False, False])

    # Build list of all organ names we need to find
    all_organ_names = set()
    for cls in target_classes:
        if cls in groups:
            # It's a group - add all members
            all_organ_names.update(groups[cls])
        else:
            # It's an individual organ
            all_organ_names.add(cls)

    logger.info(f"Looking for {len(all_organ_names)} organ labels")

    # Load model config
    logger.info("Loading model config...")
    with config_file.open("rb") as f:
        model_config = pickle.load(f)

    # Build label tree and find indices
    tree_labels = load_tree_labels()
    tree = build_label_tree(tree_labels)

    label_indices = get_label_indices(tree, list(all_organ_names))
    logger.info(f"Found {len(label_indices)} matching labels in vocabulary")

    for name, idx in sorted(label_indices.items()):
        logger.info(f"  {name}: vocabulary index {idx}")

    # Get leaf node mapping
    adjacency_matrix = model_config["adjacency_matrix"]
    leaf_nodes, original_to_pruned = get_leaf_node_info(adjacency_matrix)

    # Map label names to pruned indices
    label_to_pruned = {}
    for name, orig_idx in label_indices.items():
        if orig_idx in original_to_pruned:
            label_to_pruned[name] = original_to_pruned[orig_idx]

    logger.info(f"Mapped {len(label_to_pruned)} labels to pruned indices")

    # Load model
    logger.info("Loading model...")
    model = torch.jit.load(model_file)
    model.cuda()
    model.eval()
    torch._C._jit_set_profiling_executor(False)

    pre_processing = get_validation_transforms(
        spacing=model_config["model"]["voxel_spacing"],
        info=None,
        intensity_properties=(
            IntensityProperties(
                mean=model_config["intensity_properties"]["mean"],
                std=model_config["intensity_properties"]["std"],
            )
            if model_config["intensity_properties"] is not None
            else None
        ),
    )

    # Load input
    nifti_path, is_temp = load_input(input_path, output_dir)

    try:
        logger.info(f"Loading image: {nifti_path}")
        with allow_missing_keys_mode(pre_processing):
            example = pre_processing({"image": nifti_path})

        # Run inference
        logger.info("Running inference...")
        start_time = time.time()

        with torch.cuda.amp.autocast(), torch.no_grad():
            prediction = (
                sliding_window_inference_with_reduction(
                    inputs=example["image"].unsqueeze(0).cuda(),
                    roi_size=model_config["model"]["roi_size"],
                    sw_batch_size=2,
                    predictor=model,
                    progress=True,
                    overlap=0.5,
                    mode="gaussian",
                    cval=(
                        (-1024 - model_config["intensity_properties"]["mean"])
                        / model_config["intensity_properties"]["std"]
                        if model_config["intensity_properties"] is not None
                        else 0.0
                    ),
                    reduction_fn=partial(
                        argmax_leaves,
                        adjacency_matrix=model_config["adjacency_matrix"]
                    ),
                )
                .cpu()
                .to(torch.int64)
            )

        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f}s")

        # Apply post-processing
        postprocess = get_postprocess_transforms(output_dir=None)
        processed = postprocess({
            "pred": prediction,
            "image_meta_dict": example["image_meta_dict"]
        })
        prediction_np = processed["pred"].numpy()[0]

        logger.info(f"Prediction shape: {prediction_np.shape}")

        # Extract target classes
        logger.info("Extracting target classes...")
        output_mask = np.zeros_like(prediction_np, dtype=np.uint8)
        label_mapping = {}

        for output_value, cls_name in enumerate(target_classes, start=1):
            if cls_name in groups:
                # It's a group - merge all members
                member_names = groups[cls_name]
                logger.info(f"  Class {output_value}: '{cls_name}' (group with {len(member_names)} members)")

                for member in member_names:
                    if member in label_to_pruned:
                        pruned_idx = label_to_pruned[member]
                        mask = prediction_np == pruned_idx
                        output_mask[mask] = output_value
                        logger.info(f"    - {member}: {np.sum(mask):,} voxels")
            else:
                # Individual organ
                if cls_name in label_to_pruned:
                    pruned_idx = label_to_pruned[cls_name]
                    mask = prediction_np == pruned_idx
                    output_mask[mask] = output_value
                    logger.info(f"  Class {output_value}: '{cls_name}': {np.sum(mask):,} voxels")
                else:
                    logger.warning(f"  Class {output_value}: '{cls_name}' not found in model!")

            label_mapping[output_value] = cls_name

        # Apply flip if specified
        if any(flip_xyz):
            logger.info(f"Applying flip: {flip_xyz}")
            if flip_xyz[0]:
                output_mask = np.flip(output_mask, axis=0)
            if flip_xyz[1]:
                output_mask = np.flip(output_mask, axis=1)
            if flip_xyz[2]:
                output_mask = np.flip(output_mask, axis=2)
            output_mask = np.ascontiguousarray(output_mask)

        # Post-processing: remove small objects
        postprocess_config = user_config.get('PostProcess', {})
        if postprocess_config.get('RemoveSmallObjects', False):
            keep_n = postprocess_config.get('KeepLargestN', 1)
            logger.info(f"Removing small objects (keeping {keep_n} largest)...")
            output_mask = remove_small_objects(output_mask, keep_n=keep_n)

        # Resample to original image space
        logger.info("Resampling to original image space...")
        original_img = nib.load(nifti_path)
        original_shape = original_img.shape

        if output_mask.shape != original_shape:
            zoom_factors = [o / m for o, m in zip(original_shape, output_mask.shape)]
            output_mask = zoom(output_mask, zoom_factors, order=0, mode='nearest')

        # Save
        input_name = input_path.name if input_path.is_dir() else input_path.stem.replace('.nii', '')
        target_name = user_config.get('Target', 'segmentation').replace(' ', '_')
        output_path = output_dir / f"{input_name}_{target_name}.nii.gz"

        mask_nifti = nib.Nifti1Image(
            output_mask.astype(np.uint8),
            affine=original_img.affine,
            header=original_img.header
        )
        nib.save(mask_nifti, output_path)

        logger.info(f"Saved: {output_path}")

        # Save label info
        label_file = output_dir / f"{input_name}_{target_name}_labels.txt"
        with open(label_file, "w") as f:
            f.write(f"# {user_config.get('Target', 'Segmentation')} Labels\n")
            for value, name in sorted(label_mapping.items()):
                count = np.sum(output_mask == value)
                f.write(f"{value}: {name} ({count:,} voxels)\n")

        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("SEGMENTATION COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Config: {config_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Time: {inference_time:.2f}s")
        for value, name in sorted(label_mapping.items()):
            logger.info(f"  {value}: {name} ({np.sum(output_mask == value):,} voxels)")
        logger.info("=" * 50)

        return output_path

    finally:
        if is_temp and not keep_temp:
            nifti_path.unlink(missing_ok=True)


def main():
    parser = ArgumentParser(description="SALT Segmentation with Config File")

    parser.add_argument("--config", "-c", type=Path, required=True,
                        help="YAML config file")
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Input image (NIfTI or DICOM folder)")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Output directory")
    parser.add_argument("--model-file", type=Path,
                        default=Path("models/foobar-31/model.pt"))
    parser.add_argument("--config-file", type=Path,
                        default=Path("models/foobar-31/config.pkl"))
    parser.add_argument("--keep-temp", action="store_true")

    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    run_inference(
        config_path=args.config,
        input_path=args.input,
        output_dir=args.output,
        model_file=args.model_file,
        config_file=args.config_file,
        keep_temp=args.keep_temp,
    )


if __name__ == "__main__":
    main()
