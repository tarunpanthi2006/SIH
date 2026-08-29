"""
SatQuery Grounding Tool — Person 1 Interface
===============================================
Text-guided spatial grounding for remote sensing imagery.
Locates objects/features and returns bounding box coordinates.

Usage:
    from backend.tools.grounding import run_grounding

    result = run_grounding("path/to/satellite.png", "water body")
"""

from __future__ import annotations

import logging
from pathlib import Path

from backend.tools.interfaces import ToolResult, validate_image_path, Timer

logger = logging.getLogger(__name__)


def run_grounding(image_path: str, query: str) -> dict:
    """
    Locate a specific object or feature in a remote sensing image.

    Args:
        image_path: Path to the satellite/RS image
        query: What to locate (e.g., "water body", "buildings", "forest area")

    Returns:
        ToolResult dict with spatial evidence:
        {
            "task": "grounding",
            "model": "SatQuery-RS",
            "answer": "Water body identified in the southeast region.",
            "confidence": 0.89,
            "spatial_evidence": [
                {"type": "bbox", "coordinates": [0.6, 0.7, 0.9, 0.95]}
            ],
            "artifacts": [],
            "metadata": {...},
            "warnings": []
        }
    """
    # Validate inputs
    is_valid, error_msg = validate_image_path(image_path)
    if not is_valid:
        return ToolResult.error("grounding", "SatQuery-RS", error_msg)

    if not query or not query.strip():
        return ToolResult.error("grounding", "SatQuery-RS", "Query is empty")

    query = query.strip()
    warnings = []

    try:
        with Timer() as timer:
            from models.grounding.inference import grounding_inference

            answer, spatial_evidence, confidence = grounding_inference(
                image_path, query
            )

        # Warn if no bounding boxes were found
        if not spatial_evidence:
            warnings.append(
                f"No spatial regions identified for '{query}'. "
                f"The model's textual response may still contain useful information."
            )

        # Get model name
        try:
            from models.vqa.model import get_model
            model_name = get_model().model_name
        except Exception:
            model_name = "SatQuery-RS"

        result = ToolResult(
            task="grounding",
            model=model_name,
            answer=answer,
            confidence=confidence,
            spatial_evidence=spatial_evidence,
            artifacts=[],
            metadata={
                "image": str(Path(image_path).name),
                "query": query,
                "num_regions": len(spatial_evidence),
                "inference_time_s": round(timer.elapsed, 2),
                "base_model": "geochat-7B",
            },
            warnings=warnings,
        )

        logger.info(
            f"Grounding completed in {timer.elapsed:.2f}s — "
            f"{len(spatial_evidence)} regions, confidence: {confidence}"
        )
        return result.to_dict()

    except ImportError as e:
        return ToolResult.error(
            "grounding", "SatQuery-RS",
            f"Model dependencies not installed: {e}. "
            f"Run: pip install -r requirements.txt"
        )
    except Exception as e:
        logger.error(f"Grounding inference failed: {e}", exc_info=True)
        return ToolResult.error("grounding", "SatQuery-RS", f"Inference failed: {str(e)}")


# Allow direct testing
if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to test image")
    parser.add_argument("--query", default="water body")
    parser.add_argument("--test", action="store_true", help="Run schema test")
    args = parser.parse_args()

    if args.test:
        result = ToolResult(
            task="grounding",
            model="SatQuery-RS",
            answer="Water body identified in the southeast quadrant of the image.",
            confidence=0.89,
            spatial_evidence=[
                {"type": "bbox", "coordinates": [0.6, 0.7, 0.9, 0.95], "label": "water body"}
            ],
            artifacts=[],
            metadata={"test": True},
            warnings=[],
        ).to_dict()
        print(json.dumps(result, indent=2))
        print("✅ Schema test passed")
    elif args.image:
        result = run_grounding(args.image, args.query)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python -m backend.tools.grounding --image <path> --query <text>")
        print("       python -m backend.tools.grounding --test")
