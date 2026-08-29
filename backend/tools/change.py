"""
SatQuery Change-VQA Tool — Person 1 Interface
================================================
Bi-temporal change interpretation for remote sensing imagery.

Consumes Person 3's ChangeFormer output (change mask) and uses
the VLM to provide semantic interpretation of detected changes.

Pipeline:
    P3 ChangeFormer → change mask
                ↓
    P2 Change-VQA → "What changed?", "Where?", "How much?"

Usage:
    from backend.tools.change import run_change_vqa

    result = run_change_vqa(
        image_a="path/to/before.png",
        image_b="path/to/after.png",
        question="What changed between these two images?",
        change_mask="path/to/change_mask.png",  # from P3
    )
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from backend.tools.interfaces import ToolResult, validate_image_path, Timer

logger = logging.getLogger(__name__)


def _compute_change_statistics(change_mask_path: str) -> dict:
    """
    Compute statistics from a binary change mask produced by P3's ChangeFormer.

    Args:
        change_mask_path: Path to binary change mask (white=changed, black=unchanged)

    Returns:
        dict with change_percentage, bounding_box, centroid, pixel_count
    """
    mask = Image.open(change_mask_path).convert("L")
    mask_array = np.array(mask)

    # Threshold to binary (handle soft masks)
    binary_mask = (mask_array > 127).astype(np.uint8)

    total_pixels = binary_mask.size
    changed_pixels = int(binary_mask.sum())
    change_percentage = round(changed_pixels / max(total_pixels, 1) * 100, 2)

    # Find bounding box of changed region (normalized to [0, 1])
    h, w = binary_mask.shape
    bbox = None
    centroid = None

    if changed_pixels > 0:
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        bbox = [
            round(x_min / w, 4),
            round(y_min / h, 4),
            round((x_max + 1) / w, 4),
            round((y_max + 1) / h, 4),
        ]

        # Centroid of changed pixels
        y_coords, x_coords = np.where(binary_mask)
        centroid = [
            round(float(x_coords.mean()) / w, 4),
            round(float(y_coords.mean()) / h, 4),
        ]

    return {
        "change_percentage": change_percentage,
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "bounding_box": bbox,
        "centroid": centroid,
    }


def _create_change_overlay(
    image_path: str,
    change_mask_path: str,
    output_path: str,
    overlay_color: tuple = (255, 0, 0, 100),
) -> str:
    """
    Create a visualization overlaying the change mask on the image.

    Args:
        image_path: Path to the satellite image (before or after)
        change_mask_path: Path to the binary change mask
        output_path: Where to save the overlay image
        overlay_color: RGBA color for changed regions

    Returns:
        Path to the saved overlay image
    """
    image = Image.open(image_path).convert("RGBA")
    mask = Image.open(change_mask_path).convert("L").resize(image.size)

    # Create colored overlay
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask_array = np.array(mask)
    overlay_array = np.array(overlay)

    # Apply color to changed regions
    changed = mask_array > 127
    overlay_array[changed] = overlay_color
    overlay = Image.fromarray(overlay_array, "RGBA")

    # Composite
    result = Image.alpha_composite(image, overlay).convert("RGB")
    result.save(output_path)

    return output_path


def _build_change_prompt(
    question: str,
    change_stats: dict,
    change_mask_path: Optional[str] = None,
) -> str:
    """
    Build an enriched prompt that includes P3's change statistics
    so the VLM can provide accurate semantic interpretation.
    """
    context_parts = [
        "You are analyzing bi-temporal satellite imagery for change detection.",
        "",
    ]

    if change_stats:
        pct = change_stats.get("change_percentage", 0)
        context_parts.append(
            f"Change detection analysis shows that {pct}% of the image area has changed."
        )

        bbox = change_stats.get("bounding_box")
        if bbox:
            # Convert normalized coords to compass directions
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            ns = "northern" if cy < 0.33 else ("central" if cy < 0.66 else "southern")
            ew = "western" if cx < 0.33 else ("central" if cx < 0.66 else "eastern")
            context_parts.append(
                f"The primary area of change is in the {ns}-{ew} region of the image."
            )

    context_parts.append("")
    context_parts.append(f"Question: {question}")

    return "\n".join(context_parts)


def run_change_vqa(
    image_a: str,
    image_b: str,
    question: str,
    change_mask: Optional[str] = None,
) -> dict:
    """
    Bi-temporal change interpretation using VLM + P3's change mask.

    This function bridges P3's ChangeFormer output with P2's VLM
    to provide semantic answers about what changed.

    Args:
        image_a: Path to the "before" satellite image
        image_b: Path to the "after" satellite image
        question: Natural language question about the change
                  (e.g., "What changed?", "Is there new construction?")
        change_mask: Optional path to P3's binary change mask.
                     If None, the VLM will answer based on visual
                     comparison alone (less accurate).

    Returns:
        ToolResult dict:
        {
            "task": "change_vqa",
            "model": "SatQuery-RS",
            "answer": "New residential buildings appeared in the southeast...",
            "confidence": 0.82,
            "spatial_evidence": [
                {"type": "bbox", "coordinates": [0.6, 0.7, 0.9, 0.95], "label": "change region"}
            ],
            "artifacts": ["path/to/change_overlay.png"],
            "metadata": {
                "change_percentage": 12.3,
                "change_source": "changeformer"
            },
            "warnings": []
        }
    """
    # Validate inputs
    is_valid_a, err_a = validate_image_path(image_a)
    if not is_valid_a:
        return ToolResult.error("change_vqa", "SatQuery-RS", f"Image A: {err_a}")

    is_valid_b, err_b = validate_image_path(image_b)
    if not is_valid_b:
        return ToolResult.error("change_vqa", "SatQuery-RS", f"Image B: {err_b}")

    if not question or not question.strip():
        return ToolResult.error("change_vqa", "SatQuery-RS", "Question is empty")

    question = question.strip()
    warnings = []
    spatial_evidence = []
    artifacts = []
    change_stats = {}

    try:
        with Timer() as timer:
            # Step 1: Process change mask from P3 (if provided)
            if change_mask and os.path.exists(change_mask):
                change_stats = _compute_change_statistics(change_mask)
                logger.info(
                    f"Change mask stats: {change_stats['change_percentage']}% changed"
                )

                # Add change region as spatial evidence
                if change_stats.get("bounding_box"):
                    spatial_evidence.append({
                        "type": "bbox",
                        "coordinates": change_stats["bounding_box"],
                        "label": "change region",
                        "confidence": None,
                    })

                # Generate overlay visualization
                try:
                    overlay_dir = Path("evaluation/visualizations")
                    overlay_dir.mkdir(parents=True, exist_ok=True)
                    overlay_path = str(
                        overlay_dir / f"change_overlay_{Path(image_b).stem}.png"
                    )
                    _create_change_overlay(image_b, change_mask, overlay_path)
                    artifacts.append(overlay_path)
                except Exception as e:
                    logger.warning(f"Change overlay generation failed: {e}")
            else:
                if change_mask:
                    warnings.append(
                        f"Change mask not found at {change_mask}. "
                        f"Answering based on visual comparison of image B alone."
                    )
                else:
                    warnings.append(
                        "No change mask provided. For accurate change analysis, "
                        "run P3's ChangeFormer first and pass the mask."
                    )

            # Step 2: Build enriched prompt with change context
            enriched_question = _build_change_prompt(
                question, change_stats, change_mask
            )

            # Step 3: Run VLM inference on the "after" image
            # (The VLM analyzes the post-change image with context
            #  from the change mask statistics)
            from models.vqa.inference import vqa_inference

            answer, confidence = vqa_inference(image_b, enriched_question)

        # Get model name
        try:
            from models.vqa.model import get_model
            model_name = get_model().model_name
        except Exception:
            model_name = "SatQuery-RS"

        result = ToolResult(
            task="change_vqa",
            model=model_name,
            answer=answer,
            confidence=confidence,
            spatial_evidence=spatial_evidence,
            artifacts=artifacts,
            metadata={
                "image_before": str(Path(image_a).name),
                "image_after": str(Path(image_b).name),
                "question": question,
                "change_percentage": change_stats.get("change_percentage"),
                "changed_pixels": change_stats.get("changed_pixels"),
                "change_source": "changeformer" if change_mask else "visual_only",
                "inference_time_s": round(timer.elapsed, 2),
            },
            warnings=warnings,
        )

        logger.info(
            f"Change-VQA completed in {timer.elapsed:.2f}s — "
            f"answer: '{answer[:80]}...' confidence: {confidence}"
        )
        return result.to_dict()

    except ImportError as e:
        return ToolResult.error(
            "change_vqa", "SatQuery-RS",
            f"Model dependencies not installed: {e}. "
            f"Run: pip install -r requirements.txt"
        )
    except Exception as e:
        logger.error(f"Change-VQA inference failed: {e}", exc_info=True)
        return ToolResult.error(
            "change_vqa", "SatQuery-RS", f"Inference failed: {str(e)}"
        )


# Allow direct testing
if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-a", help="Path to 'before' image")
    parser.add_argument("--image-b", help="Path to 'after' image")
    parser.add_argument("--mask", help="Path to P3's change mask", default=None)
    parser.add_argument(
        "--question",
        default="What changes are visible between the two time periods?",
    )
    parser.add_argument("--test", action="store_true", help="Run schema test")
    args = parser.parse_args()

    if args.test:
        result = ToolResult(
            task="change_vqa",
            model="SatQuery-RS",
            answer="New residential construction has appeared in the southeast quadrant.",
            confidence=0.82,
            spatial_evidence=[
                {
                    "type": "bbox",
                    "coordinates": [0.6, 0.7, 0.9, 0.95],
                    "label": "change region",
                }
            ],
            artifacts=[],
            metadata={
                "change_percentage": 12.3,
                "change_source": "changeformer",
                "test": True,
            },
            warnings=[],
        ).to_dict()
        print(json.dumps(result, indent=2))
        print("[OK] Schema test passed")
    elif args.image_a and args.image_b:
        result = run_change_vqa(
            args.image_a, args.image_b, args.question, args.mask
        )
        print(json.dumps(result, indent=2))
    else:
        print(
            "Usage: python -m backend.tools.change "
            "--image-a <before> --image-b <after> --question <text> [--mask <path>]"
        )
        print("       python -m backend.tools.change --test")
