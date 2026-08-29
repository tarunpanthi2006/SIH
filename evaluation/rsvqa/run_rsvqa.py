"""
RSVQA Evaluation Runner
=========================
Downloads RSVQA-LR/HR datasets and evaluates SatQuery-RS.

RSVQA: Remote sensing VQA with presence, comparison, counting questions.

Usage:
    python -m evaluation.rsvqa.run_rsvqa --max-samples 500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RSVQA_LR_ZENODO = "https://doi.org/10.5281/zenodo.6344333"
RSVQA_HR_ZENODO = "https://doi.org/10.5281/zenodo.6344366"
RSVQA_GITHUB = "https://github.com/sylvainlobry/rsvqa"


def evaluate_on_rsvqa(
    data_path: str | None = None,
    max_samples: int = 500,
    output_dir: str = "evaluation/rsvqa",
) -> dict:
    """
    Run SatQuery-RS evaluation on RSVQA dataset.

    Metrics: Overall Accuracy (OA), Average Accuracy (AA),
             per-question-type accuracy.
    """
    from training.finetuning.evaluate import compute_vqa_accuracy
    from collections import Counter

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if data_path is None or not os.path.exists(data_path):
        logger.warning(
            f"RSVQA data not found. Please download from:\n"
            f"  LR: {RSVQA_LR_ZENODO}\n"
            f"  HR: {RSVQA_HR_ZENODO}\n"
            f"  Code: {RSVQA_GITHUB}\n"
            f"Then run: python -m evaluation.rsvqa.run_rsvqa --data <path>"
        )
        return {"benchmark": "RSVQA", "error": "Data not found"}

    # Load RSVQA data
    with open(data_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # RSVQA format: {"questions": [...], "answers": [...]}
        questions = data.get("questions", [])
        answers = data.get("answers", [])
    else:
        questions = data
        answers = None

    logger.info(f"Loaded {len(questions)} RSVQA questions")

    if max_samples < len(questions):
        import random
        random.seed(42)
        indices = random.sample(range(len(questions)), max_samples)
        questions = [questions[i] for i in indices]
        if answers:
            answers = [answers[i] for i in indices]

    results = {"benchmark": "RSVQA", "model": "SatQuery-RS"}

    # Run inference
    predictions = []
    references = []
    question_types = []

    for idx, q in enumerate(questions):
        image_path = q.get("image", q.get("img_id", ""))
        question_text = q.get("question", q.get("input", ""))
        q_type = q.get("type", q.get("question_type", "unknown"))

        if answers:
            reference = answers[idx].get("answer", answers[idx].get("output", ""))
        else:
            reference = q.get("answer", q.get("output", ""))

        if not os.path.exists(str(image_path)):
            continue

        try:
            from models.vqa.inference import vqa_inference
            pred, _ = vqa_inference(str(image_path), question_text)
            predictions.append(pred)
            references.append(reference)
            question_types.append(q_type)
        except Exception as e:
            logger.warning(f"Inference failed: {e}")

    if predictions:
        # Overall accuracy
        results["overall"] = compute_vqa_accuracy(predictions, references)

        # Per-type accuracy
        type_preds: dict[str, list] = {}
        type_refs: dict[str, list] = {}
        for pred, ref, qtype in zip(predictions, references, question_types):
            type_preds.setdefault(qtype, []).append(pred)
            type_refs.setdefault(qtype, []).append(ref)

        results["per_type"] = {}
        for qtype in type_preds:
            results["per_type"][qtype] = compute_vqa_accuracy(
                type_preds[qtype], type_refs[qtype]
            )

        # Average Accuracy (mean of per-type accuracies)
        type_accuracies = [
            v["exact_match"] for v in results["per_type"].values()
        ]
        if type_accuracies:
            results["average_accuracy"] = round(
                sum(type_accuracies) / len(type_accuracies), 2
            )

        logger.info(f"RSVQA Results: {json.dumps(results, indent=2)}")

    # Save results
    results_path = output_path / "rsvqa_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate on RSVQA")
    parser.add_argument("--data", type=str, default=None, help="Path to RSVQA JSON")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="evaluation/rsvqa")
    args = parser.parse_args()

    evaluate_on_rsvqa(args.data, args.max_samples, args.output_dir)


if __name__ == "__main__":
    main()
