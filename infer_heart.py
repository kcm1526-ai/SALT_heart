#!/usr/bin/env python3
"""
Heart Segmentation Inference Script for SALT

This script performs heart segmentation from CT images (DICOM folder or NIfTI format).
It segments the heart organ with 5 structures:
- Heart myocardium
- Left atrium
- Left ventricle
- Right atrium
- Right ventricle

Usage:
    # From DICOM folder (folder containing .dcm files):
    python infer_heart.py --input /path/to/dicom_folder --output /path/to/output

    # From NIfTI file:
    python infer_heart.py --input /path/to/image.nii.gz --output /path/to/output

    # Binary mask (all heart structures combined as 1):
    python infer_heart.py --input /path/to/dicom_folder --output /path/to/output --binary

Author: Generated based on SALT framework
"""

import logging
import pickle
import time
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
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


# Heart structure labels (5 core heart structures only)
HEART_LABELS = {
    "heart_myocardium": {"file_index": 57, "output_value": 1},
    "heart_atrium_left": {"file_index": 58, "output_value": 2},
    "heart_ventricle_left": {"file_index": 59, "output_value": 3},
    "heart_atrium_right": {"file_index": 60, "output_value": 4},
    "heart_ventricle_right": {"file_index": 61, "output_value": 5},
}


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


def get_leaf_to_original_mapping(adjacency_matrix: np.ndarray) -> Dict[int, int]:
    """Create mapping from leaf node indices to original label indices."""
    leave_nodes = np.where(adjacency_matrix[1:, 1:].sum(axis=1) == 0)[0]
    return {i: int(leave_nodes[i]) for i in range(len(leave_nodes))}


def dicom_to_nifti(dicom_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    Convert DICOM folder to NIfTI format.

    Args:
        dicom_path: Path to folder containing .dcm files
        output_path: Optional output path for NIfTI file

    Returns:
        Path to the created NIfTI file
    """
    logger.info(f"Converting DICOM from {dicom_path} to NIfTI...")

    reader = sitk.ImageSeriesReader()
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_path))

    if len(series_ids) == 0:
        # Try finding .dcm files directly
        dcm_files = list(dicom_path.glob("*.dcm")) + list(dicom_path.glob("*.DCM"))
        if len(dcm_files) == 0:
            all_files = [f for f in dicom_path.iterdir() if f.is_file()]
            raise ValueError(
                f"No DICOM files found in {dicom_path}. "
                f"Found {len(all_files)} files but none recognized as DICOM."
            )
        dcm_files = sorted(dcm_files, key=lambda x: x.name)
        dicom_names = [str(f) for f in dcm_files]
        logger.info(f"Found {len(dicom_names)} DICOM files")
    else:
        logger.info(f"Found {len(series_ids)} DICOM series")
        dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            str(dicom_path), series_ids[0]
        )
        logger.info(f"Using series with {len(dicom_names)} slices")

    reader.SetFileNames(dicom_names)
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
    leaf_to_original: Dict[int, int],
    binary: bool = False,
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Extract heart structures from full body segmentation."""
    original_to_leaf = {v: k for k, v in leaf_to_original.items()}

    heart_mask = np.zeros_like(prediction, dtype=np.uint8)
    label_mapping = {}

    for label_name, info in HEART_LABELS.items():
        original_idx = info["file_index"]
        output_value = info["output_value"] if not binary else 1

        if original_idx in original_to_leaf:
            leaf_idx = original_to_leaf[original_idx]
            mask_locations = prediction == leaf_idx

            if np.any(mask_locations):
                heart_mask[mask_locations] = output_value
                label_mapping[output_value] = label_name
                logger.info(f"  Found {label_name}: {np.sum(mask_locations)} voxels")

    if binary:
        label_mapping = {1: "heart"}

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

    # Load model
    logger.info("Loading model...")
    with config_file.open("rb") as f:
        config = pickle.load(f)

    model = torch.jit.load(model_file)
    model.cuda()
    model.eval()
    torch._C._jit_set_profiling_executor(False)

    leaf_to_original = get_leaf_to_original_mapping(config["adjacency_matrix"])

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
        prediction = prediction[0]

        heart_mask, label_mapping = extract_heart_mask(
            prediction,
            leaf_to_original,
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
        """
    )

    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="Input DICOM folder or NIfTI file"
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
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

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")
    if not args.model_file.exists():
        raise FileNotFoundError(f"Model not found: {args.model_file}")
    if not args.config_file.exists():
        raise FileNotFoundError(f"Config not found: {args.config_file}")

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
