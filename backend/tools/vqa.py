"""
SatQuery VQA Tool — Person 1 Interface
========================================
Single-image Visual Question Answering for remote sensing imagery.

This is the clean interface that Person 1's tool registry calls.

Usage:
    from backend.tools.vqa import run_vqa

    result = run_vqa("path/to/satellite.png", "What land cover is present?")
"""

from __future__ import annotations

import logging
from pathlib import Path

from backend.tools.interfaces import ToolResult, validate_image_path, Timer

logger = logging.getLogger(__name__)


def run_vqa(image_path: str, question: str) -> dict:
    """
    Single-image Visual Question Answering.

    Called by Person 1's tool registry. Loads the SatQuery-RS model
    (or base GeoChat-7B if no adapter is available) and runs inference.

    Args:
        image_path: Path to the satellite/RS image
        question: Natural language question about the image

    Returns:
        ToolResult dict conforming to the SatQuery JSON contract:
        {
            "task": "vqa",
            "model": "SatQuery-RS",
            "answer": "...",
            "confidence": 0.91,
            "spatial_evidence": [],
            "artifacts": [],
            "metadata": {...},
            "warnings": []
        }
    """
    # Validate inputs
    is_valid, error_msg = validate_image_path(image_path)
    if not is_valid:
        return ToolResult.error("vqa", "SatQuery-RS", error_msg)

    if not question or not question.strip():
        return ToolResult.error("vqa", "SatQuery-RS", "Question is empty")

    question = question.strip()
    warnings = []

    try:
        with Timer() as timer:
            from models.vqa.inference import vqa_inference

            answer, confidence = vqa_inference(image_path, question)

        # Get model name from the loaded model
        try:
            from models.vqa.model import get_model
            model_name = get_model().model_name
        except Exception:
            model_name = "SatQuery-RS"

        result = ToolResult(
            task="vqa",
            model=model_name,
            answer=answer,
            confidence=confidence,
            spatial_evidence=[],
            artifacts=[],
            metadata={
                "image": str(Path(image_path).name),
                "question": question,
                "inference_time_s": round(timer.elapsed, 2),
                "base_model": "geochat-7B",
            },
            warnings=warnings,
        )

        logger.info(
            f"VQA completed in {timer.elapsed:.2f}s — "
            f"answer: '{answer[:80]}...' confidence: {confidence}"
        )
        return result.to_dict()

    except ImportError as e:
        return ToolResult.error(
            "vqa", "SatQuery-RS",
            f"Model dependencies not installed: {e}. "
            f"Run: pip install -r requirements.txt"
        )
    except Exception as e:
        logger.error(f"VQA inference failed: {e}", exc_info=True)
        return ToolResult.error("vqa", "SatQuery-RS", f"Inference failed: {str(e)}")


# Allow direct testing
if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to test image")
    parser.add_argument("--question", default="What is visible in this satellite image?")
    parser.add_argument("--test", action="store_true", help="Run with mock data")
    args = parser.parse_args()

    if args.test:
        # Schema validation test (no model needed)
        result = ToolResult(
            task="vqa",
            model="SatQuery-RS",
            answer="The image shows agricultural fields with mixed vegetation.",
            confidence=0.87,
            spatial_evidence=[],
            artifacts=[],
            metadata={"test": True},
            warnings=[],
        ).to_dict()
        print(json.dumps(result, indent=2))
        print("✅ Schema test passed")
    elif args.image:
        result = run_vqa(args.image, args.question)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python -m backend.tools.vqa --image <path> --question <text>")
        print("       python -m backend.tools.vqa --test")
