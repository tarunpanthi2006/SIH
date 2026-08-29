"""
SatQuery — Change Detection Model Bridge (P3 Interface)
=========================================================
Placeholder model loader for P3's ChangeFormer.

P3 owns the actual ChangeFormer model and weights.
This module provides a standardized interface for P2 to call P3's
change detection without needing to understand ChangeFormer internals.

When P3's code is available, this module acts as a thin wrapper.
When P3's code is NOT available, it falls back to image differencing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def detect_changes_fallback(
    image_before: str,
    image_after: str,
    threshold: float = 30.0,
    output_mask_path: Optional[str] = None,
) -> str:
    """
    Simple image differencing fallback when P3's ChangeFormer is not available.

    This is NOT production quality. It is a basic absolute difference + threshold
    to allow P2 to test the Change-VQA pipeline independently of P3.

    For production, P3's ChangeFormer should be used instead.

    Args:
        image_before: Path to T1 image
        image_after: Path to T2 image
        threshold: Pixel difference threshold (0-255)
        output_mask_path: Where to save the generated mask

    Returns:
        Path to the generated change mask
    """
    logger.warning(
        "Using fallback image differencing. "
        "For accurate results, use P3's ChangeFormer."
    )

    img_a = Image.open(image_before).convert("RGB")
    img_b = Image.open(image_after).convert("RGB").resize(img_a.size)

    arr_a = np.array(img_a, dtype=np.float32)
    arr_b = np.array(img_b, dtype=np.float32)

    # Absolute difference across channels
    diff = np.abs(arr_a - arr_b).mean(axis=2)

    # Threshold
    mask = (diff > threshold).astype(np.uint8) * 255

    # Save mask
    if output_mask_path is None:
        output_mask_path = str(
            Path(image_after).parent / f"change_mask_{Path(image_after).stem}.png"
        )

    Image.fromarray(mask, mode="L").save(output_mask_path)
    logger.info(f"Fallback change mask saved to: {output_mask_path}")

    return output_mask_path


def run_changeformer(
    image_before: str,
    image_after: str,
    output_mask_path: Optional[str] = None,
) -> str:
    """
    Run P3's ChangeFormer model on a bi-temporal image pair.

    Attempts to import P3's module. Falls back to image differencing
    if P3's code is not available.

    Args:
        image_before: Path to T1 image
        image_after: Path to T2 image
        output_mask_path: Where to save the change mask

    Returns:
        Path to the generated change mask
    """
    try:
        # Try to import P3's ChangeFormer implementation
        from models.change_detection.changeformer import predict_change

        logger.info("Using P3's ChangeFormer for change detection")
        mask_path = predict_change(
            image_before, image_after, output_path=output_mask_path
        )
        return mask_path

    except ImportError:
        logger.info(
            "P3's ChangeFormer not available, using fallback image differencing"
        )
        return detect_changes_fallback(
            image_before, image_after, output_mask_path=output_mask_path
        )
