"""
CDVQA (Change Detection VQA) Evaluation
==========================================
Evaluates the P2+P3 Change-VQA pipeline on bi-temporal VQA benchmarks.

CDVQA requires:
    - Bi-temporal image pairs (before/after)
    - Change masks from P3's ChangeFormer
    - Ground-truth questions and answers

Usage:
    python -m evaluation.cdvqa.evaluate --data <path> --max-samples 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from evaluation.cdvqa.metrics import (
    compute_change_vqa_accuracy,
    compute_change_detection_f1,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_change_vqa(
    data_path: str | None = None,
    max_samples: int = 100,
    output_dir: str = "evaluation/cdvqa",
) -> dict:
    """
    Evaluate Change-VQA pipeline on a CDVQA-format dataset.

    Expected data format (JSON list):
    [
        {
            "image_before": "path/to/t1.png",
            "image_after": "path/to/t2.png",
            "change_mask": "path/to/mask.png",  // optional
            "question": "What changed?",
            "answer": "New buildings appeared.",
            "type": "semantic"  // semantic | spatial | yes_no
        },
        ...
    ]
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if data_path is None or not os.path.exists(data_path):
        logger.warning(
            "CDVQA data not found. To evaluate Change-VQA:\n"
            "  1. Prepare bi-temporal image pairs with change masks\n"
            "  2. Create a JSON file with the format shown in the docstring\n"
            "  3. Run: python -m evaluation.cdvqa.evaluate --data <path>"
        )
        return {"benchmark": "CDVQA", "error": "Data not found"}

    # Load data
    with open(data_path, "r") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} CDVQA samples")

    if max_samples < len(data):
        import random
        random.seed(42)
        data = random.sample(data, max_samples)

    results = {"benchmark": "CDVQA", "model": "SatQuery-RS + ChangeFormer"}

    predictions = []
    references = []
    question_types = []

    for sample in data:
        image_before = sample.get("image_before", "")
        image_after = sample.get("image_after", "")
        change_mask = sample.get("change_mask")
        question = sample.get("question", "")
        reference = sample.get("answer", "")
        q_type = sample.get("type", "semantic")

        # Skip if images don't exist
        if not os.path.exists(image_before) or not os.path.exists(image_after):
            continue

        try:
            from backend.tools.change import run_change_vqa

            result = run_change_vqa(
                image_a=image_before,
                image_b=image_after,
                question=question,
                change_mask=change_mask,
            )

            pred = result.get("answer", "")
            predictions.append(pred)
            references.append(reference)
            question_types.append(q_type)

        except Exception as e:
            logger.warning(f"Change-VQA failed: {e}")

    if predictions:
        results["overall"] = compute_change_vqa_accuracy(predictions, references)

        # Per-type breakdown
        type_preds: dict[str, list] = {}
        type_refs: dict[str, list] = {}
        for pred, ref, qtype in zip(predictions, references, question_types):
            type_preds.setdefault(qtype, []).append(pred)
            type_refs.setdefault(qtype, []).append(ref)

        results["per_type"] = {}
        for qtype in type_preds:
            results["per_type"][qtype] = compute_change_vqa_accuracy(
                type_preds[qtype], type_refs[qtype]
            )

        logger.info(f"CDVQA Results: {json.dumps(results, indent=2)}")
    else:
        results["error"] = "No valid samples processed"
        logger.warning("No valid samples were processed")

    # Save results
    results_path = output_path / "cdvqa_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Change-VQA")
    parser.add_argument("--data", type=str, default=None, help="Path to CDVQA JSON")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="evaluation/cdvqa")
    args = parser.parse_args()

    evaluate_change_vqa(args.data, args.max_samples, args.output_dir)


if __name__ == "__main__":
    main()
