#!/usr/bin/env python3
"""
Heart Segmentation Inference Script for SALT

This script performs heart segmentation from CT images (DICOM or NIfTI format).
It uses the pre-trained SALT model to segment heart structures including:
- Heart myocardium
- Left atrium
- Right atrium
- Left ventricle
- Right ventricle
- Pericardium (optional)
- Aorta (thoracic, pass pericardium) (optional)
- Pulmonary artery (pass pericardium) (optional)

Usage:
    # From DICOM folder:
    python infer_heart.py --input /path/to/dicom_folder --output /path/to/output

    # From NIfTI file:
    python infer_heart.py --input /path/to/image.nii.gz --output /path/to/output

    # Binary mask (all heart structures combined):
    python infer_heart.py --input /path/to/input --output /path/to/output --binary

    # Include pericardium and vessels:
    python infer_heart.py --input /path/to/input --output /path/to/output --include-pericardium --include-vessels

Author: Generated based on SALT framework
"""

import logging
import pickle
import time
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
from monai.transforms import Compose, SaveImage
from monai.transforms.utils import allow_missing_keys_mode

from salt.input_pipeline import (
    IntensityProperties,
    get_validation_transforms,
)
from salt.utils.inference import sliding_window_inference_with_reduction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Heart-related label definitions based on saros+totalseg labels
# Format: (label_index_in_file, label_name, is_core_heart)
HEART_LABELS = {
    "heart_myocardium": {"file_index": 57, "output_value": 1, "is_core": True},
    "heart_atrium_left": {"file_index": 58, "output_value": 2, "is_core": True},
    "heart_ventricle_left": {"file_index": 59, "output_value": 3, "is_core": True},
    "heart_atrium_right": {"file_index": 60, "output_value": 4, "is_core": True},
    "heart_ventricle_right": {"file_index": 61, "output_value": 5, "is_core": True},
    "pericardium": {"file_index": 7, "output_value": 6, "is_core": False},
    "aorta_thoracica_pass_pericardium": {"file_index": 118, "output_value": 7, "is_core": False},
    "pulmonary_artery_pass_pericardium": {"file_index": 120, "output_value": 8, "is_core": False},
}


def argmax_leaves(
    inputs: torch.Tensor,
    adjacency_matrix: np.ndarray,
    dim: int = 1,
    pruned: bool = True,
) -> torch.Tensor:
    """
    Compute argmax over leaf nodes in the label tree.

    Args:
        inputs: Model output probabilities [B, C, D, H, W]
        adjacency_matrix: Tree structure matrix
        dim: Dimension to compute argmax over
        pruned: If True, return indices relative to leaf nodes only

    Returns:
        Tensor with predicted leaf node indices
    """
    leave_nodes = np.where(adjacency_matrix[1:, 1:].sum(axis=1) == 0)[0]
    indices = np.arange(adjacency_matrix.shape[0] - 1, dtype=np.int32)
    indices = indices[leave_nodes]
    y_pred_leaves = inputs[:, leave_nodes]
    y_pred_leave_idx = torch.argmax(y_pred_leaves, axis=dim)
    if pruned:
        return y_pred_leave_idx
    return torch.tensor(indices).to(inputs.device)[y_pred_leave_idx]


def get_leaf_to_original_mapping(adjacency_matrix: np.ndarray) -> Dict[int, int]:
    """
    Create mapping from leaf node indices to original label indices.

    Args:
        adjacency_matrix: Tree structure matrix

    Returns:
        Dictionary mapping leaf index to original label index
    """
    leave_nodes = np.where(adjacency_matrix[1:, 1:].sum(axis=1) == 0)[0]
    return {i: int(leave_nodes[i]) for i in range(len(leave_nodes))}


def dicom_to_nifti(dicom_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    Convert DICOM series to NIfTI format.

    Args:
        dicom_path: Path to DICOM folder
        output_path: Optional output path for NIfTI file

    Returns:
        Path to the created NIfTI file
    """
    logger.info(f"Converting DICOM from {dicom_path} to NIfTI...")

    # Read DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_path))

    if len(dicom_names) == 0:
        raise ValueError(f"No DICOM files found in {dicom_path}")

    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # Determine output path
    if output_path is None:
        output_path = dicom_path.parent / f"{dicom_path.name}_converted.nii.gz"

    # Write NIfTI
    sitk.WriteImage(image, str(output_path))
    logger.info(f"DICOM converted and saved to {output_path}")

    return output_path


def load_input(input_path: Path, temp_dir: Optional[Path] = None) -> Tuple[Path, bool]:
    """
    Load input file, converting from DICOM if necessary.

    Args:
        input_path: Path to input file or DICOM folder
        temp_dir: Temporary directory for converted files

    Returns:
        Tuple of (nifti_path, is_temporary)
    """
    if input_path.is_dir():
        # Assume it's a DICOM folder
        if temp_dir is None:
            temp_dir = input_path.parent
        nifti_path = dicom_to_nifti(
            input_path,
            temp_dir / f"{input_path.name}_temp.nii.gz"
        )
        return nifti_path, True
    elif input_path.suffix in [".gz", ".nii"]:
        # Already NIfTI
        return input_path, False
    else:
        raise ValueError(f"Unsupported input format: {input_path}")


def extract_heart_mask(
    prediction: np.ndarray,
    leaf_to_original: Dict[int, int],
    include_pericardium: bool = False,
    include_vessels: bool = False,
    binary: bool = False,
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Extract heart structures from full body segmentation.

    Args:
        prediction: Full segmentation prediction array
        leaf_to_original: Mapping from leaf indices to original label indices
        include_pericardium: Include pericardium in the mask
        include_vessels: Include aorta and pulmonary artery
        binary: If True, combine all structures into binary mask

    Returns:
        Tuple of (heart_mask, label_mapping)
    """
    # Create reverse mapping (original index -> leaf index)
    original_to_leaf = {v: k for k, v in leaf_to_original.items()}

    # Initialize output mask
    heart_mask = np.zeros_like(prediction, dtype=np.uint8)
    label_mapping = {}

    for label_name, info in HEART_LABELS.items():
        # Skip non-core labels if not requested
        if not info["is_core"]:
            if label_name == "pericardium" and not include_pericardium:
                continue
            if "aorta" in label_name or "pulmonary" in label_name:
                if not include_vessels:
                    continue

        original_idx = info["file_index"]
        output_value = info["output_value"] if not binary else 1

        # Find the leaf index that corresponds to this original index
        if original_idx in original_to_leaf:
            leaf_idx = original_to_leaf[original_idx]
            mask_locations = prediction == leaf_idx

            if np.any(mask_locations):
                heart_mask[mask_locations] = output_value
                label_mapping[output_value] = label_name
                logger.info(f"  Found {label_name}: {np.sum(mask_locations)} voxels")

    if binary:
        label_mapping = {1: "heart_combined"}

    return heart_mask, label_mapping


def save_mask(
    mask: np.ndarray,
    reference_image: nib.Nifti1Image,
    output_path: Path,
    label_mapping: Dict[int, str],
) -> None:
    """
    Save segmentation mask as NIfTI file with proper header.

    Args:
        mask: Segmentation mask array
        reference_image: Reference NIfTI image for header/affine
        output_path: Output file path
        label_mapping: Mapping of label values to names
    """
    # Create NIfTI image with same affine and header as reference
    mask_nifti = nib.Nifti1Image(
        mask.astype(np.uint8),
        affine=reference_image.affine,
        header=reference_image.header.copy()
    )

    # Update data type in header
    mask_nifti.header.set_data_dtype(np.uint8)

    # Save
    nib.save(mask_nifti, output_path)
    logger.info(f"Heart mask saved to {output_path}")

    # Save label mapping
    label_file = output_path.parent / f"{output_path.stem.replace('.nii', '')}_labels.txt"
    with open(label_file, "w") as f:
        f.write("# Heart Segmentation Labels\n")
        f.write("# Value: Label Name\n")
        for value, name in sorted(label_mapping.items()):
            f.write(f"{value}: {name}\n")
    logger.info(f"Label mapping saved to {label_file}")


def run_inference(
    input_path: Path,
    output_dir: Path,
    model_file: Path,
    config_file: Path,
    include_pericardium: bool = False,
    include_vessels: bool = False,
    binary: bool = False,
    keep_temp: bool = False,
) -> Path:
    """
    Run heart segmentation inference.

    Args:
        input_path: Input DICOM folder or NIfTI file
        output_dir: Output directory
        model_file: Path to TorchScript model
        config_file: Path to config pickle file
        include_pericardium: Include pericardium in output
        include_vessels: Include vessels in output
        binary: Create binary mask (all structures combined)
        keep_temp: Keep temporary files

    Returns:
        Path to output mask file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and config
    logger.info("Loading model and configuration...")
    with config_file.open("rb") as f:
        config = pickle.load(f)

    model = torch.jit.load(model_file)
    model.cuda()
    model.eval()

    # Disable JIT profiling for speed
    torch._C._jit_set_profiling_executor(False)

    # Get leaf to original mapping
    leaf_to_original = get_leaf_to_original_mapping(config["adjacency_matrix"])

    # Setup preprocessing
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

    # Load input (convert DICOM if needed)
    nifti_path, is_temp = load_input(input_path, output_dir)

    try:
        # Load and preprocess
        logger.info(f"Loading image from {nifti_path}...")
        with allow_missing_keys_mode(pre_processing):
            example = pre_processing({"image": nifti_path})

        # Load reference image for saving
        reference_image = nib.load(nifti_path)

        # Run inference
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

        # Extract heart mask
        logger.info("Extracting heart structures...")
        prediction = prediction[0]  # Remove batch dimension

        heart_mask, label_mapping = extract_heart_mask(
            prediction,
            leaf_to_original,
            include_pericardium=include_pericardium,
            include_vessels=include_vessels,
            binary=binary,
        )

        # Determine output filename
        input_name = input_path.stem
        if input_name.endswith("_temp"):
            input_name = input_name[:-5]
        output_name = f"{input_name}_heart_mask.nii.gz"
        output_path = output_dir / output_name

        # Save mask
        save_mask(heart_mask, reference_image, output_path, label_mapping)

        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info("HEART SEGMENTATION COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Inference time: {inference_time:.2f}s")
        logger.info(f"Total heart voxels: {np.sum(heart_mask > 0)}")
        logger.info("Label values:")
        for value, name in sorted(label_mapping.items()):
            count = np.sum(heart_mask == value)
            logger.info(f"  {value}: {name} ({count} voxels)")
        logger.info("=" * 50)

        return output_path

    finally:
        # Cleanup temporary files
        if is_temp and not keep_temp:
            nifti_path.unlink(missing_ok=True)
            logger.info("Cleaned up temporary files")


def main():
    parser = ArgumentParser(
        description="Heart Segmentation Inference using SALT model",
        epilog="""
Examples:
  # Basic usage with DICOM:
  python infer_heart.py --input /path/to/dicom_folder --output /path/to/output

  # Basic usage with NIfTI:
  python infer_heart.py --input /path/to/image.nii.gz --output /path/to/output

  # Binary mask (all heart as single label):
  python infer_heart.py --input /path/to/input --output /path/to/output --binary

  # Include pericardium and vessels:
  python infer_heart.py --input /path/to/input --output /path/to/output --include-pericardium --include-vessels
        """
    )

    # Required arguments
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input DICOM folder or NIfTI file (.nii.gz)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for heart mask"
    )

    # Model arguments
    parser.add_argument(
        "--model-file",
        type=Path,
        default=Path("models/foobar-31/model.pt"),
        help="Path to TorchScript model file (default: models/foobar-31/model.pt)"
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("models/foobar-31/config.pkl"),
        help="Path to config pickle file (default: models/foobar-31/config.pkl)"
    )

    # Output options
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Output binary mask (all heart structures as value 1)"
    )
    parser.add_argument(
        "--include-pericardium",
        action="store_true",
        help="Include pericardium in the output mask"
    )
    parser.add_argument(
        "--include-vessels",
        action="store_true",
        help="Include aorta and pulmonary artery (pericardium portions) in output"
    )

    # Other options
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files (converted NIfTI from DICOM)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    if not args.model_file.exists():
        raise FileNotFoundError(
            f"Model file not found: {args.model_file}\n"
            "Please ensure you have downloaded the model files.\n"
            "If using Git LFS, run: git lfs pull"
        )

    if not args.config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {args.config_file}\n"
            "Please ensure you have downloaded the config files.\n"
            "If using Git LFS, run: git lfs pull"
        )

    # Run inference
    output_path = run_inference(
        input_path=args.input,
        output_dir=args.output,
        model_file=args.model_file,
        config_file=args.config_file,
        include_pericardium=args.include_pericardium,
        include_vessels=args.include_vessels,
        binary=args.binary,
        keep_temp=args.keep_temp,
    )

    print(f"\nHeart mask saved to: {output_path}")


if __name__ == "__main__":
    main()
