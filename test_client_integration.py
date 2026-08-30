"""
SatQuery-RS — Integration Test & Example for Person 1
======================================================
This script demonstrates how Person 1 (Agent developer) should call
the SatQuery-RS model server endpoints robustly using the client.

Usage:
    # First, make sure the model server is running:
    # python -m backend.serve --host 0.0.0.0 --port 8100

    # Then run this test script:
    python test_client_integration.py
"""

import json
import logging
import os
from pathlib import Path

# Important: Person 1 can just configure this environment variable
# to point to the Lightning AI instance (e.g. https://8100-studio.lightning.ai)
os.environ["SATQUERY_SERVER_URL"] = "http://localhost:8100"

# Import the drop-in convenience functions from the client
from backend.client import (
    run_vqa,
    run_caption,
    run_grounding,
    run_change_vqa,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Agent-Mock")

def print_result(name: str, result: dict):
    print(f"\n{'='*50}\n[ {name} ] Result:\n{'='*50}")
    print(json.dumps(result, indent=2))
    
    # Showcase error handling
    if result.get("metadata", {}).get("error"):
        print("❌ This request failed (expected if server is offline or image is missing).")
        print(f"Warning: {result.get('warnings')}")
    else:
        print(f"✅ Success! Answer: {result.get('answer')}")

def main():
    logger.info("Starting integration test for Person 1's Agent")
    
    # We will use a dummy file for testing the API schema/errors
    # (Since we don't have a real satellite image here, we'll create a 1x1 pixel image)
    dummy_image = "dummy_test_image.png"
    if not Path(dummy_image).exists():
        from PIL import Image
        Image.new("RGB", (10, 10)).save(dummy_image)

    # 1. Test VQA (Visual Question Answering)
    logger.info("Agent Tool Call: run_vqa()")
    res_vqa = run_vqa(
        image_path=dummy_image,
        question="Are there any buildings in this image?"
    )
    print_result("VQA", res_vqa)

    # 2. Test Captioning
    logger.info("Agent Tool Call: run_caption()")
    res_caption = run_caption(
        image_path=dummy_image,
        instruction="Describe the land cover types visible."
    )
    print_result("Captioning", res_caption)

    # 3. Test Grounding (Bounding Boxes)
    logger.info("Agent Tool Call: run_grounding()")
    res_grounding = run_grounding(
        image_path=dummy_image,
        query="water body"
    )
    print_result("Grounding", res_grounding)

    # 4. Test Change-VQA (The P2 + P3 Bridge)
    logger.info("Agent Tool Call: run_change_vqa()")
    res_change = run_change_vqa(
        image_a=dummy_image,
        image_b=dummy_image,
        question="What type of development appeared between these two periods?"
    )
    print_result("Change-VQA", res_change)

    # Clean up dummy image
    if Path(dummy_image).exists():
        Path(dummy_image).unlink()

if __name__ == "__main__":
    main()
