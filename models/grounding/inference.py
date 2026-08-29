"""
SatQuery-RS — Grounding Inference Engine
==========================================
Text-guided spatial grounding for remote sensing imagery.

GeoChat outputs bounding boxes as coordinate tokens in its
generated text, e.g.:
    "The water body is located at [0.32, 0.45, 0.78, 0.82]"

This module parses those coordinates and returns structured
SpatialEvidence objects.

Usage:
    from models.grounding.inference import grounding_inference

    answer, bboxes, confidence = grounding_inference(
        "path/to/image.png",
        "water body"
    )
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ============================================================
# Bounding box parsing
# ============================================================

# Regex patterns for extracting bounding boxes from GeoChat output
BBOX_PATTERNS = [
    # [x1, y1, x2, y2] format (normalized 0-1)
    r'\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]',
    # {x1, y1, x2, y2} format
    r'\{\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\}',
    # (x1, y1, x2, y2) format
    r'\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)',
]


def parse_bounding_boxes(text: str) -> list[list[float]]:
    """
    Extract bounding box coordinates from model-generated text.

    Handles multiple formats and normalizes to [0, 1] range.

    Returns:
        List of [x1, y1, x2, y2] bounding boxes, each normalized to [0, 1]
    """
    bboxes = []

    for pattern in BBOX_PATTERNS:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                coords = [float(c) for c in match]

                # If coordinates are in pixel range (>1), normalize
                # GeoChat typically outputs [0, 1] but sometimes [0, 100] or [0, 1000]
                max_coord = max(coords)
                if max_coord > 1.0:
                    if max_coord <= 100:
                        coords = [c / 100.0 for c in coords]
                    elif max_coord <= 1000:
                        coords = [c / 1000.0 for c in coords]
                    else:
                        # Skip obviously wrong coordinates
                        logger.warning(f"Skipping bbox with large coords: {coords}")
                        continue

                # Validate: x1 < x2, y1 < y2, all in [0, 1]
                x1, y1, x2, y2 = coords
                if (0 <= x1 <= 1 and 0 <= y1 <= 1 and
                    0 <= x2 <= 1 and 0 <= y2 <= 1 and
                    x1 < x2 and y1 < y2):
                    bboxes.append(coords)
                elif (0 <= x1 <= 1 and 0 <= y1 <= 1 and
                      0 <= x2 <= 1 and 0 <= y2 <= 1):
                    # Swap if needed
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    if x1 != x2 and y1 != y2:
                        bboxes.append([x1, y1, x2, y2])

            except (ValueError, IndexError):
                continue

    return bboxes


def format_spatial_evidence(bboxes: list[list[float]], label: str = "") -> list[dict]:
    """
    Convert parsed bounding boxes to SpatialEvidence dicts.
    """
    evidence = []
    for bbox in bboxes:
        evidence.append({
            "type": "bbox",
            "coordinates": [round(c, 4) for c in bbox],
            "label": label if label else None,
        })
    return evidence


# ============================================================
# Grounding inference
# ============================================================

def grounding_inference(
    image_path: str,
    query: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
) -> tuple[str, list[dict], float]:
    """
    Run text-guided grounding on a single image.

    Args:
        image_path: Path to the satellite/RS image
        query: What to locate (e.g., "water body", "buildings")
        max_new_tokens: Maximum tokens to generate
        temperature: Generation temperature

    Returns:
        (answer_text, spatial_evidence_list, confidence_score)
    """
    from models.vqa.model import get_model
    from models.vqa.inference import load_and_preprocess_image, format_prompt

    vlm = get_model()

    # Format as grounding instruction
    instruction = (
        f"Locate the {query} in this image and provide its "
        f"bounding box coordinates as [x1, y1, x2, y2] where "
        f"coordinates are normalized between 0 and 1."
    )

    logger.info(f"Grounding inference: '{query}' on {Path(image_path).name}")

    # Preprocess image
    image_tensor = load_and_preprocess_image(image_path, vlm.image_processor)

    # Format prompt
    input_ids = format_prompt(instruction, vlm.tokenizer, has_image=True)

    # Generate
    answer, confidence = vlm.generate(
        input_ids=input_ids,
        images=image_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=False,
    )

    # Parse bounding boxes from generated text
    bboxes = parse_bounding_boxes(answer)
    spatial_evidence = format_spatial_evidence(bboxes, label=query)

    if not spatial_evidence:
        logger.info(f"No bounding boxes found in grounding output: '{answer[:100]}'")
    else:
        logger.info(f"Found {len(spatial_evidence)} bounding boxes for '{query}'")

    return answer, spatial_evidence, confidence


# CLI test
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--query", default="water body", help="What to locate")
    args = parser.parse_args()

    answer, evidence, conf = grounding_inference(args.image, args.query)
    print(f"Query: {args.query}")
    print(f"Answer: {answer}")
    print(f"Evidence: {evidence}")
    print(f"Confidence: {conf}")
