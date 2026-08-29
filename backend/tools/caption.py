"""
SatQuery Caption Tool — Person 1 Interface
=============================================
Single-image scene description / captioning for remote sensing imagery.

Usage:
    from backend.tools.caption import run_caption

    result = run_caption("path/to/satellite.png")
"""

from __future__ import annotations

import logging
from pathlib import Path

from backend.tools.interfaces import ToolResult, validate_image_path, Timer

logger = logging.getLogger(__name__)


def run_caption(image_path: str, instruction: str | None = None) -> dict:
    """
    Generate a detailed scene description of a remote sensing image.

    Args:
        image_path: Path to the satellite/RS image
        instruction: Optional custom captioning instruction.
                     Default: "Describe this satellite image in detail."

    Returns:
        ToolResult dict:
        {
            "task": "caption",
            "model": "SatQuery-RS",
            "answer": "A satellite image showing agricultural fields...",
            "confidence": 0.88,
            "spatial_evidence": [],
            "artifacts": [],
            "metadata": {...},
            "warnings": []
        }
    """
    # Validate inputs
    is_valid, error_msg = validate_image_path(image_path)
    if not is_valid:
        return ToolResult.error("caption", "SatQuery-RS", error_msg)

    warnings = []

    try:
        with Timer() as timer:
            from models.vqa.inference import caption_inference

            caption, confidence = caption_inference(
                image_path,
                instruction=instruction,
            )

        # Get model name
        try:
            from models.vqa.model import get_model
            model_name = get_model().model_name
        except Exception:
            model_name = "SatQuery-RS"

        result = ToolResult(
            task="caption",
            model=model_name,
            answer=caption,
            confidence=confidence,
            spatial_evidence=[],
            artifacts=[],
            metadata={
                "image": str(Path(image_path).name),
                "instruction": instruction or "Describe this satellite image in detail.",
                "inference_time_s": round(timer.elapsed, 2),
                "base_model": "geochat-7B",
            },
            warnings=warnings,
        )

        logger.info(
            f"Caption completed in {timer.elapsed:.2f}s — "
            f"'{caption[:80]}...' confidence: {confidence}"
        )
        return result.to_dict()

    except ImportError as e:
        return ToolResult.error(
            "caption", "SatQuery-RS",
            f"Model dependencies not installed: {e}. "
            f"Run: pip install -r requirements.txt"
        )
    except Exception as e:
        logger.error(f"Caption inference failed: {e}", exc_info=True)
        return ToolResult.error("caption", "SatQuery-RS", f"Inference failed: {str(e)}")


# Allow direct testing
if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to test image")
    parser.add_argument("--test", action="store_true", help="Run schema test")
    args = parser.parse_args()

    if args.test:
        result = ToolResult(
            task="caption",
            model="SatQuery-RS",
            answer="An aerial view of a coastal region with sandy beaches and dense urban development.",
            confidence=0.85,
            spatial_evidence=[],
            artifacts=[],
            metadata={"test": True},
            warnings=[],
        ).to_dict()
        print(json.dumps(result, indent=2))
        print("✅ Schema test passed")
    elif args.image:
        result = run_caption(args.image)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python -m backend.tools.caption --image <path>")
        print("       python -m backend.tools.caption --test")
