"""
VRSBench Evaluation Runner
============================
Downloads VRSBench and evaluates SatQuery-RS on its VQA, captioning,
and grounding subsets.

VRSBench: 29,614 images, 123,221 VQA pairs, 29,614 captions, 52,472 grounding refs.

Usage:
    python -m evaluation.vrsbench.run_vrsbench --max-samples 500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VRSBENCH_HF_DATASET = "xiang709/VRSBench"
VRSBENCH_GITHUB = "https://github.com/lx709/VRSBench"


def download_vrsbench(output_dir: Path) -> Path:
    """Download VRSBench dataset from HuggingFace."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        logger.info(f"Downloading VRSBench from HuggingFace: {VRSBENCH_HF_DATASET}")
        dataset = load_dataset(VRSBENCH_HF_DATASET, split="test")

        # Save locally
        data_path = output_dir / "vrsbench_test.json"
        dataset.to_json(str(data_path))
        logger.info(f"VRSBench saved to: {data_path}")
        return data_path

    except Exception as e:
        logger.error(f"Failed to download VRSBench: {e}")
        logger.info(f"Manual download: {VRSBENCH_GITHUB}")
        raise


def evaluate_on_vrsbench(
    data_path: str | None = None,
    max_samples: int = 500,
    output_dir: str = "evaluation/vrsbench",
) -> dict:
    """
    Run SatQuery-RS evaluation on VRSBench.
    """
    from training.finetuning.evaluate import compute_vqa_accuracy, compute_caption_metrics

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download if needed
    if data_path is None:
        data_path = str(download_vrsbench(output_path))

    # Load VRSBench data
    with open(data_path, "r") as f:
        # Handle both JSON and JSONL
        content = f.read().strip()
        if content.startswith("["):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.split("\n") if line.strip()]

    logger.info(f"Loaded {len(data)} VRSBench samples")

    # Limit samples
    if max_samples < len(data):
        import random
        random.seed(42)
        data = random.sample(data, max_samples)

    # Separate by task type
    vqa_data = [d for d in data if d.get("task", "") == "vqa" or "question" in d]
    caption_data = [d for d in data if d.get("task", "") == "caption" or "caption" in d]

    results = {"benchmark": "VRSBench", "model": "SatQuery-RS"}

    # VQA evaluation
    if vqa_data:
        logger.info(f"Evaluating VQA on {len(vqa_data)} samples...")
        predictions = []
        references = []

        for sample in vqa_data:
            image_path = sample.get("image", sample.get("image_path", ""))
            question = sample.get("question", sample.get("input", ""))
            reference = sample.get("answer", sample.get("output", ""))

            if not os.path.exists(image_path):
                continue

            try:
                from models.vqa.inference import vqa_inference
                pred, _ = vqa_inference(image_path, question)
                predictions.append(pred)
                references.append(reference)
            except Exception as e:
                logger.warning(f"VQA inference failed: {e}")

        if predictions:
            results["vqa"] = compute_vqa_accuracy(predictions, references)
            logger.info(f"VRSBench VQA: {results['vqa']}")

    # Caption evaluation
    if caption_data:
        logger.info(f"Evaluating captioning on {len(caption_data)} samples...")
        predictions = []
        references = []

        for sample in caption_data:
            image_path = sample.get("image", sample.get("image_path", ""))
            reference = sample.get("caption", sample.get("output", ""))

            if not os.path.exists(image_path):
                continue

            try:
                from models.vqa.inference import caption_inference
                pred, _ = caption_inference(image_path)
                predictions.append(pred)
                references.append(reference)
            except Exception as e:
                logger.warning(f"Caption inference failed: {e}")

        if predictions:
            results["caption"] = compute_caption_metrics(predictions, references)
            logger.info(f"VRSBench Caption: {results['caption']}")

    # Save results
    results_path = output_path / "vrsbench_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate on VRSBench")
    parser.add_argument("--data", type=str, default=None, help="Path to VRSBench data")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="evaluation/vrsbench")
    args = parser.parse_args()

    evaluate_on_vrsbench(args.data, args.max_samples, args.output_dir)


if __name__ == "__main__":
    main()
