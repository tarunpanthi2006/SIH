"""
SatQuery — Change Detection Inference Bridge
===============================================
Bridge module between P3's ChangeFormer output and P2's VLM interpretation.

This module handles the loading and processing of change detection results
from P3 (ChangeFormer, BIT, etc.) and passes them to the VLM for semantic
interpretation.

Usage:
    from models.change.inference import interpret_change

    result = interpret_change(
        image_before="path/to/t1.png",
        image_after="path/to/t2.png",
        change_mask="path/to/mask.png",
        question="What type of change occurred?",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def load_change_mask(mask_path: str) -> np.ndarray:
    """
    Load a binary change mask produced by P3's change detection model.

    Supports:
    - Binary PNG masks (0/255)
    - Soft probability masks (0.0-1.0 saved as 8-bit)
    - Multi-class change masks (for future CDVQA)

    Args:
        mask_path: Path to the change mask image

    Returns:
        Binary numpy array (H, W) where 1=changed, 0=unchanged
    """
    mask = Image.open(mask_path).convert("L")
    mask_array = np.array(mask, dtype=np.float32) / 255.0

    # Binarize with threshold
    binary = (mask_array > 0.5).astype(np.uint8)

    logger.info(
        f"Loaded change mask: {mask_array.shape}, "
        f"{binary.sum()} changed pixels ({binary.mean() * 100:.1f}%)"
    )
    return binary


def extract_change_regions(
    binary_mask: np.ndarray,
    min_area_ratio: float = 0.001,
) -> list[dict]:
    """
    Extract distinct change regions from a binary mask using
    connected component analysis.

    Args:
        binary_mask: Binary mask (H, W) where 1=changed
        min_area_ratio: Minimum region area as fraction of total image

    Returns:
        List of region dicts with bbox, area, centroid
    """
    try:
        from scipy import ndimage
    except ImportError:
        # Fallback: treat the entire changed area as one region
        logger.warning("scipy not installed — using single-region fallback")
        return _fallback_single_region(binary_mask)

    labeled_array, num_features = ndimage.label(binary_mask)
    h, w = binary_mask.shape
    total_area = h * w
    min_area = total_area * min_area_ratio

    regions = []
    for region_id in range(1, num_features + 1):
        region_mask = labeled_array == region_id
        area = int(region_mask.sum())

        if area < min_area:
            continue

        # Bounding box
        rows = np.any(region_mask, axis=1)
        cols = np.any(region_mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Centroid
        y_coords, x_coords = np.where(region_mask)

        regions.append({
            "bbox": [
                round(x_min / w, 4),
                round(y_min / h, 4),
                round((x_max + 1) / w, 4),
                round((y_max + 1) / h, 4),
            ],
            "area_pixels": area,
            "area_fraction": round(area / total_area, 4),
            "centroid": [
                round(float(x_coords.mean()) / w, 4),
                round(float(y_coords.mean()) / h, 4),
            ],
        })

    # Sort by area (largest first)
    regions.sort(key=lambda r: r["area_pixels"], reverse=True)

    logger.info(f"Extracted {len(regions)} change regions from mask")
    return regions


def _fallback_single_region(binary_mask: np.ndarray) -> list[dict]:
    """Fallback when scipy is not available."""
    h, w = binary_mask.shape
    total_area = h * w
    changed = int(binary_mask.sum())

    if changed == 0:
        return []

    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    y_coords, x_coords = np.where(binary_mask)

    return [{
        "bbox": [
            round(x_min / w, 4),
            round(y_min / h, 4),
            round((x_max + 1) / w, 4),
            round((y_max + 1) / h, 4),
        ],
        "area_pixels": changed,
        "area_fraction": round(changed / total_area, 4),
        "centroid": [
            round(float(x_coords.mean()) / w, 4),
            round(float(y_coords.mean()) / h, 4),
        ],
    }]


def interpret_change(
    image_before: str,
    image_after: str,
    change_mask: str,
    question: str = "What changes are visible between the two time periods?",
) -> dict:
    """
    Full pipeline: load P3's mask → extract regions → VLM interpretation.

    This is the core function that bridges P3 and P2.

    Args:
        image_before: Path to the T1 (before) image
        image_after: Path to the T2 (after) image
        change_mask: Path to P3's binary change mask
        question: Natural language question about the change

    Returns:
        dict with answer, regions, change statistics
    """
    # Load and analyze mask
    binary_mask = load_change_mask(change_mask)
    regions = extract_change_regions(binary_mask)

    h, w = binary_mask.shape
    change_pct = round(float(binary_mask.mean()) * 100, 2)

    # Build context-rich prompt for VLM
    context = (
        f"You are analyzing bi-temporal satellite imagery for change detection.\n"
        f"Change detection analysis shows that {change_pct}% of the area has changed.\n"
    )

    if regions:
        context += f"There are {len(regions)} distinct change regions detected.\n"
        largest = regions[0]
        cx = (largest["bbox"][0] + largest["bbox"][2]) / 2
        cy = (largest["bbox"][1] + largest["bbox"][3]) / 2
        ns = "northern" if cy < 0.33 else ("central" if cy < 0.66 else "southern")
        ew = "western" if cx < 0.33 else ("central" if cx < 0.66 else "eastern")
        context += (
            f"The largest change region is in the {ns}-{ew} part of the image, "
            f"covering {largest['area_fraction'] * 100:.1f}% of the total area.\n"
        )

    context += f"\nQuestion: {question}"

    # VLM inference on the after image with context
    from models.vqa.inference import vqa_inference

    answer, confidence = vqa_inference(image_after, context)

    return {
        "answer": answer,
        "confidence": confidence,
        "change_percentage": change_pct,
        "num_regions": len(regions),
        "regions": regions,
    }
