"""
SatQuery-RS — VQA & Captioning Inference Engine
=================================================
Core inference functions for single-image VQA and captioning.

These functions are called by the tool wrappers in backend/tools/.

Usage:
    from models.vqa.inference import vqa_inference, caption_inference

    answer, confidence = vqa_inference("path/to/image.png", "What is visible?")
    caption, confidence = caption_inference("path/to/image.png")
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger(__name__)


def load_and_preprocess_image(
    image_path: str,
    image_processor,
) -> torch.Tensor:
    """
    Load an image and preprocess it for the CLIP vision encoder.

    Args:
        image_path: Path to the image file
        image_processor: CLIPImageProcessor instance

    Returns:
        Preprocessed image tensor
    """
    image = Image.open(image_path).convert("RGB")

    # Process using CLIP image processor
    processed = image_processor.preprocess(image, return_tensors="pt")
    pixel_values = processed["pixel_values"]

    # Move to appropriate device/dtype
    if torch.cuda.is_available():
        pixel_values = pixel_values.to(dtype=torch.float16, device="cuda")
    else:
        pixel_values = pixel_values.to(dtype=torch.float16)

    return pixel_values


def format_prompt(
    instruction: str,
    tokenizer,
    has_image: bool = True,
) -> torch.Tensor:
    """
    Format a prompt in GeoChat/LLaVA conversation style.

    GeoChat uses the Vicuna-v1.5 conversation template:
    USER: <image>\n{instruction}
    ASSISTANT:

    Args:
        instruction: The question or instruction text
        tokenizer: The model's tokenizer
        has_image: Whether the prompt includes an image

    Returns:
        Tokenized input_ids tensor
    """
    if has_image:
        prompt = f"USER: <image>\n{instruction}\nASSISTANT:"
    else:
        prompt = f"USER: {instruction}\nASSISTANT:"

    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=2048,
    ).input_ids

    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")

    return input_ids


def vqa_inference(
    image_path: str,
    question: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
) -> tuple[str, float]:
    """
    Run Visual Question Answering on a single image.

    Args:
        image_path: Path to the satellite/RS image
        question: Natural language question about the image
        max_new_tokens: Maximum tokens to generate
        temperature: Generation temperature

    Returns:
        (answer_text, confidence_score)
    """
    from models.vqa.model import get_model

    vlm = get_model()

    logger.info(f"VQA inference: '{question}' on {Path(image_path).name}")

    # Preprocess image
    image_tensor = load_and_preprocess_image(image_path, vlm.image_processor)

    # Format prompt
    input_ids = format_prompt(question, vlm.tokenizer, has_image=True)

    # Generate
    answer, confidence = vlm.generate(
        input_ids=input_ids,
        images=image_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=False,
    )

    logger.info(f"VQA result: '{answer[:100]}...' (confidence: {confidence})")
    return answer, confidence


def caption_inference(
    image_path: str,
    instruction: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> tuple[str, float]:
    """
    Generate a scene description/caption for a satellite image.

    Args:
        image_path: Path to the satellite/RS image
        instruction: Optional custom captioning instruction
        max_new_tokens: Maximum tokens to generate
        temperature: Generation temperature

    Returns:
        (caption_text, confidence_score)
    """
    from models.vqa.model import get_model

    vlm = get_model()

    if instruction is None:
        instruction = "Describe this satellite image in detail."

    logger.info(f"Caption inference on {Path(image_path).name}")

    # Preprocess image
    image_tensor = load_and_preprocess_image(image_path, vlm.image_processor)

    # Format prompt
    input_ids = format_prompt(instruction, vlm.tokenizer, has_image=True)

    # Generate
    caption, confidence = vlm.generate(
        input_ids=input_ids,
        images=image_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=False,
    )

    logger.info(f"Caption result: '{caption[:100]}...' (confidence: {confidence})")
    return caption, confidence


def batch_vqa_inference(
    image_paths: list[str],
    questions: list[str],
    max_new_tokens: int = 256,
) -> list[tuple[str, float]]:
    """
    Run VQA on multiple image-question pairs.
    Processes sequentially (batch=1) to minimize VRAM usage.

    Used primarily for evaluation (VRSBench, RSVQA).
    """
    results = []
    for image_path, question in zip(image_paths, questions):
        try:
            answer, confidence = vqa_inference(
                image_path, question, max_new_tokens
            )
            results.append((answer, confidence))
        except Exception as e:
            logger.error(f"Failed on {image_path}: {e}")
            results.append(("", 0.0))

    return results


# CLI test
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--question", default="What is visible in this image?")
    parser.add_argument("--mode", choices=["vqa", "caption"], default="vqa")
    args = parser.parse_args()

    if args.mode == "vqa":
        answer, conf = vqa_inference(args.image, args.question)
        print(f"Question: {args.question}")
        print(f"Answer: {answer}")
        print(f"Confidence: {conf}")
    else:
        caption, conf = caption_inference(args.image)
        print(f"Caption: {caption}")
        print(f"Confidence: {conf}")
