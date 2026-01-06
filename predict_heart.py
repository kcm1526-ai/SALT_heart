import logging
import pickle
import time
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path

import numpy as np
import torch
from monai.transforms.utils import allow_missing_keys_mode

from salt.input_pipeline import (
    IntensityProperties,
    get_postprocess_transforms,
    get_validation_transforms,
)
from salt.core.label_tree import LabelTree
from salt.utils.inference import sliding_window_inference_with_reduction

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Heart structure names
HEART_LABEL_NAMES = [
    "heart_myocardium",
    "heart_atrium_left",
    "heart_ventricle_left",
    "heart_atrium_right",
    "heart_ventricle_right",
]

# Path to tree-labels file
TREE_LABELS_FILE = Path(__file__).parent / "labels" / "saros+totalseg" / "tree-labels.txt"


def load_tree_labels(tree_labels_file: Path = TREE_LABELS_FILE):
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


def build_label_tree(tree_labels):
    """Build a LabelTree from tree labels."""
    tree = LabelTree()
    for label in tree_labels:
        if label is not None:
            tree.add(*label)
    tree.optimize()
    return tree


def get_heart_pruned_indices(adjacency_matrix: np.ndarray):
    """Get pruned leaf indices for heart labels."""
    # Build label tree
    tree_labels = load_tree_labels()
    tree = build_label_tree(tree_labels)
    vocabulary = tree.labels

    # Find heart label original indices
    heart_original_indices = {}
    for idx, label_tuple in enumerate(vocabulary):
        label_name = label_tuple[-1] if label_tuple else ""
        if label_name in HEART_LABEL_NAMES:
            heart_original_indices[label_name] = idx

    # Get leaf nodes and mapping
    leaf_nodes = np.where(adjacency_matrix[1:, 1:].sum(axis=1) == 0)[0]
    original_to_pruned = {int(leaf_nodes[i]): i for i in range(len(leaf_nodes))}

    # Get pruned indices for heart
    heart_pruned_indices = {}
    for name, orig_idx in heart_original_indices.items():
        if orig_idx in original_to_pruned:
            heart_pruned_indices[name] = original_to_pruned[orig_idx]
            logger.info(f"  {name}: original={orig_idx} -> pruned={original_to_pruned[orig_idx]}")

    return heart_pruned_indices


def argmax_leaves(
    inputs: torch.Tensor,
    adjacency_matrix: np.ndarray,
    dim: int = 1,
    pruned: bool = True,
) -> torch.Tensor:
    """Original argmax_leaves - unchanged."""
    leave_nodes = np.where(adjacency_matrix[1:, 1:].sum(axis=1) == 0)[0]
    indices = np.arange(adjacency_matrix.shape[0] - 1, dtype=np.int32)
    indices = indices[leave_nodes]
    y_pred_leaves = inputs[:, leave_nodes]
    y_pred_leave_idx = torch.argmax(y_pred_leaves, axis=dim)
    if pruned:
        return y_pred_leave_idx
    return torch.tensor(indices).to(inputs.device)[y_pred_leave_idx]


def extract_heart_mask(prediction: np.ndarray, heart_pruned_indices: dict) -> np.ndarray:
    """Extract only heart structures from prediction.

    Args:
        prediction: Full prediction with pruned leaf indices
        heart_pruned_indices: Dict mapping heart label name to pruned index

    Returns:
        Heart mask with values 1-5 for heart structures, 0 elsewhere
    """
    heart_mask = np.zeros_like(prediction, dtype=np.uint8)

    # Map: pruned_index -> output_value (1-5)
    label_order = ["heart_myocardium", "heart_atrium_left", "heart_ventricle_left",
                   "heart_atrium_right", "heart_ventricle_right"]

    for output_val, name in enumerate(label_order, start=1):
        if name in heart_pruned_indices:
            pruned_idx = heart_pruned_indices[name]
            mask = prediction == pruned_idx
            heart_mask[mask] = output_val
            voxel_count = np.sum(mask)
            if voxel_count > 0:
                logger.info(f"  {name}: {voxel_count:,} voxels (value={output_val})")

    return heart_mask


def main(args: Namespace) -> None:
    logger.info("Loading model...")
    with (args.config_file).open("rb") as ifile:
        config = pickle.load(ifile)

    model = torch.jit.load(args.model_file)

    # Get heart pruned indices
    logger.info("Finding heart label indices...")
    heart_pruned_indices = get_heart_pruned_indices(config["adjacency_matrix"])

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
    model.cuda()
    model.eval()

    torch._C._jit_set_profiling_executor(False)

    if args.data_dir.is_dir():
        inputs = sorted(args.data_dir.glob("*.nii.gz"))
    else:
        inputs = [args.data_dir]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for i, input_path in enumerate(inputs, start=1):
        logger.info(f"{i}/{len(inputs)}: Loading image {input_path.name}...")
        with allow_missing_keys_mode(pre_processing):
            example = pre_processing({"image": input_path})

        logger.info(f"Computing model output for {input_path.name}...")
        start = time.time()

        with torch.cuda.amp.autocast(), torch.no_grad():
            pred = (
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
                        argmax_leaves, adjacency_matrix=config["adjacency_matrix"]
                    ),
                )
                .cpu()
                .to(torch.int64)
            )

        inference_time = time.time() - start
        logger.info(f"Inference completed in {inference_time:.2f}s")

        # Apply post-processing
        logger.info("Applying post-processing...")
        postprocess = get_postprocess_transforms(output_dir=None)
        processed = postprocess({
            "pred": pred,
            "image_meta_dict": example["image_meta_dict"]
        })
        prediction_np = processed["pred"].numpy()[0]  # Remove channel dim

        # Extract heart mask
        logger.info("Extracting heart structures...")
        heart_mask = extract_heart_mask(prediction_np, heart_pruned_indices)

        total_heart = np.sum(heart_mask > 0)
        logger.info(f"Total heart voxels: {total_heart:,}")

        # Save heart mask
        import nibabel as nib
        from scipy.ndimage import zoom

        # Load original to get shape
        original_img = nib.load(input_path)
        original_shape = original_img.shape

        # Resample to original space
        if heart_mask.shape != original_shape:
            zoom_factors = [o / m for o, m in zip(original_shape, heart_mask.shape)]
            heart_mask = zoom(heart_mask, zoom_factors, order=0, mode='nearest')

        # Save
        output_name = input_path.name.replace(".nii.gz", "_heart.nii.gz")
        output_path = args.output_dir / output_name

        mask_nifti = nib.Nifti1Image(
            heart_mask.astype(np.uint8),
            affine=original_img.affine,
            header=original_img.header
        )
        nib.save(mask_nifti, output_path)

        logger.info(f"Saved: {output_path}")
        logger.info("=" * 50)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-file",
        type=Path,
        default=Path("models/foobar-31/model.pt"),
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("models/foobar-31/config.pkl"),
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    main(args)
