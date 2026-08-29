"""
SatQuery-RS — Base vs Adapted Model Evaluation
=================================================
Evaluates GeoChat-7B (base) vs SatQuery-RS (adapted) on held-out data.

Metrics:
- VQA: Accuracy (exact match, relaxed match)
- Captioning: BLEU-4, METEOR, CIDEr
- Grounding: IoU@0.5 (if bounding boxes available)

Usage:
    python -m training.finetuning.evaluate \
        --test-data datasets/bigearthnet/processed/test.json \
        --max-samples 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def compute_vqa_accuracy(predictions: list[str], references: list[str]) -> dict:
    """
    Compute VQA accuracy metrics.
    - Exact match: prediction == reference (case-insensitive)
    - Relaxed match: reference is contained in prediction (case-insensitive)
    """
    exact = 0
    relaxed = 0
    total = len(predictions)

    for pred, ref in zip(predictions, references):
        pred_clean = pred.strip().lower()
        ref_clean = ref.strip().lower()

        if pred_clean == ref_clean:
            exact += 1
            relaxed += 1
        elif ref_clean in pred_clean or pred_clean in ref_clean:
            relaxed += 1

    return {
        "exact_match": round(exact / max(total, 1) * 100, 2),
        "relaxed_match": round(relaxed / max(total, 1) * 100, 2),
        "total": total,
    }


def compute_caption_metrics(predictions: list[str], references: list[str]) -> dict:
    """
    Compute captioning metrics: BLEU-4, METEOR.
    Falls back gracefully if nltk is not available.
    """
    metrics = {}

    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk

        # Ensure punkt tokenizer is available
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)

        smoother = SmoothingFunction().method1
        bleu_scores = []

        for pred, ref in zip(predictions, references):
            ref_tokens = nltk.word_tokenize(ref.lower())
            pred_tokens = nltk.word_tokenize(pred.lower())

            if len(ref_tokens) > 0 and len(pred_tokens) > 0:
                score = sentence_bleu(
                    [ref_tokens], pred_tokens,
                    smoothing_function=smoother
                )
                bleu_scores.append(score)

        if bleu_scores:
            metrics["bleu4"] = round(sum(bleu_scores) / len(bleu_scores) * 100, 2)

    except ImportError:
        logger.warning("nltk not installed — skipping BLEU computation")

    try:
        from nltk.translate.meteor_score import meteor_score as nltk_meteor
        import nltk

        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)

        meteor_scores = []
        for pred, ref in zip(predictions, references):
            if pred.strip() and ref.strip():
                score = nltk_meteor([ref.lower()], pred.lower())
                meteor_scores.append(score)

        if meteor_scores:
            metrics["meteor"] = round(sum(meteor_scores) / len(meteor_scores) * 100, 2)

    except (ImportError, Exception) as e:
        logger.warning(f"METEOR computation failed: {e}")

    metrics["total"] = len(predictions)
    return metrics


def compute_grounding_iou(
    pred_bboxes_list: list[list[list[float]]],
    ref_bboxes_list: list[list[list[float]]],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute IoU-based grounding accuracy using the new metrics.py."""
    from evaluation.metrics import compute_iou
    
    hits = 0
    total = 0

    for preds, refs in zip(pred_bboxes_list, ref_bboxes_list):
        if not refs:
            continue
        total += len(refs)
        
        # Simple greedy matching for demonstration
        for ref in refs:
            best_iou = 0.0
            for pred in preds:
                score = compute_iou(pred, ref)
                if score > best_iou:
                    best_iou = score
                    
            if best_iou >= iou_threshold:
                hits += 1

    return {
        f"iou@{iou_threshold}": round(hits / max(total, 1) * 100, 2),
        "total": total,
    }


def evaluate_model(
    test_data_path: str,
    max_samples: int = 200,
    use_base: bool = False,
) -> dict:
    """
    Evaluate a model on held-out test data.

    Args:
        test_data_path: Path to test JSON file
        max_samples: Maximum samples to evaluate
        use_base: If True, evaluate base model (no LoRA adapter)
    """
    from models.vqa.inference import vqa_inference, caption_inference
    from models.grounding.inference import grounding_inference, parse_bounding_boxes

    # Load test data
    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    if max_samples < len(test_data):
        import random
        random.seed(42)
        test_data = random.sample(test_data, max_samples)

    model_label = "base" if use_base else "adapted"
    logger.info(f"Evaluating {model_label} model on {len(test_data)} samples")

    # Separate by task type
    vqa_samples = []
    caption_samples = []
    grounding_samples = []

    for sample in test_data:
        task_type = sample.get("metadata", {}).get("task_type", "unknown")
        if task_type in ("binary", "mcq"):
            vqa_samples.append(sample)
        elif task_type == "captioning":
            caption_samples.append(sample)
        elif task_type == "bounding box":
            grounding_samples.append(sample)

    results = {"model": model_label, "total_samples": len(test_data)}

    # VQA evaluation
    if vqa_samples:
        logger.info(f"Evaluating VQA on {len(vqa_samples)} samples...")
        vqa_preds = []
        vqa_refs = []
        for sample in vqa_samples[:max_samples]:
            image_path = sample.get("image", "")
            question = sample["conversations"][0]["value"].replace("<image>\n", "")
            reference = sample["conversations"][1]["value"]

            if os.path.exists(image_path):
                try:
                    pred, _ = vqa_inference(image_path, question)
                    vqa_preds.append(pred)
                    vqa_refs.append(reference)
                except Exception as e:
                    logger.warning(f"VQA failed: {e}")

        if vqa_preds:
            results["vqa"] = compute_vqa_accuracy(vqa_preds, vqa_refs)
            logger.info(f"VQA results: {results['vqa']}")

    # Caption evaluation
    if caption_samples:
        logger.info(f"Evaluating captioning on {len(caption_samples)} samples...")
        cap_preds = []
        cap_refs = []
        for sample in caption_samples[:max_samples]:
            image_path = sample.get("image", "")
            reference = sample["conversations"][1]["value"]

            if os.path.exists(image_path):
                try:
                    pred, _ = caption_inference(image_path)
                    cap_preds.append(pred)
                    cap_refs.append(reference)
                except Exception as e:
                    logger.warning(f"Caption failed: {e}")

        if cap_preds:
            results["caption"] = compute_caption_metrics(cap_preds, cap_refs)
            logger.info(f"Caption results: {results['caption']}")

    # Grounding evaluation & Visualization
    if grounding_samples:
        logger.info(f"Evaluating grounding on {len(grounding_samples)} samples...")
        from evaluation.visualize import draw_grounding_boxes
        
        grounding_preds = []
        grounding_refs = []
        
        vis_dir = Path("evaluation/visualizations")
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, sample in enumerate(grounding_samples[:max_samples]):
            image_path = sample.get("image", "")
            question = sample["conversations"][0]["value"].replace("<image>\n", "")
            reference = sample["conversations"][1]["value"]
            
            if os.path.exists(image_path):
                try:
                    # Grounding inference returns the text answer with bounding boxes
                    pred, _ = grounding_inference(image_path, question)
                    pred_boxes = parse_bounding_boxes(pred)
                    ref_boxes = parse_bounding_boxes(reference)
                    
                    grounding_preds.append(pred_boxes)
                    grounding_refs.append(ref_boxes)
                    
                    # Generate a visualization for the first 5 samples!
                    if idx < 5 and pred_boxes:
                        out_file = vis_dir / f"{model_label}_grounding_{idx}.png"
                        draw_grounding_boxes(
                            image_path=image_path,
                            boxes=pred_boxes,
                            labels=["AI Prediction"] * len(pred_boxes),
                            output_path=out_file
                        )
                        logger.info(f"Saved visualization: {out_file}")
                        
                except Exception as e:
                    logger.warning(f"Grounding failed: {e}")

        if grounding_preds:
            results["grounding"] = compute_grounding_iou(grounding_preds, grounding_refs)
            logger.info(f"Grounding results: {results['grounding']}")

    return results


def compare_base_vs_adapted(test_data_path: str, max_samples: int = 200) -> dict:
    """
    Run evaluation for both base and adapted models, then compare.
    """
    logger.info("=" * 60)
    logger.info("BASE vs ADAPTED MODEL COMPARISON")
    logger.info("=" * 60)

    # Evaluate base model (temporarily disable LoRA)
    os.environ["LORA_ADAPTER_PATH"] = "__none__"
    logger.info("\n--- Evaluating BASE model (GeoChat-7B) ---")
    base_results = evaluate_model(test_data_path, max_samples, use_base=True)

    # Reset model singleton for adapted model
    from models.vqa.model import SatQueryVLM
    if SatQueryVLM._instance:
        SatQueryVLM._instance.unload()
        SatQueryVLM._instance = None

    # Evaluate adapted model
    os.environ.pop("LORA_ADAPTER_PATH", None)
    logger.info("\n--- Evaluating ADAPTED model (SatQuery-RS) ---")
    adapted_results = evaluate_model(test_data_path, max_samples, use_base=False)

    comparison = {
        "base": base_results,
        "adapted": adapted_results,
    }

    # Print comparison table
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<30} {'Base':>12} {'Adapted':>12} {'Delta':>12}")
    logger.info("-" * 66)

    for task in ["vqa", "caption", "grounding"]:
        if task in base_results and task in adapted_results:
            for metric, base_val in base_results[task].items():
                if metric == "total":
                    continue
                adapted_val = adapted_results[task].get(metric, 0)
                if isinstance(base_val, (int, float)) and isinstance(adapted_val, (int, float)):
                    delta = adapted_val - base_val
                    sign = "+" if delta >= 0 else ""
                    logger.info(
                        f"  {task}/{metric:<24} {base_val:>12.2f} {adapted_val:>12.2f} {sign}{delta:>11.2f}"
                    )

    logger.info("=" * 60)

    # Save comparison
    output_path = Path("evaluation/comparison_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Results saved to: {output_path}")

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs adapted model")
    parser.add_argument(
        "--test-data",
        type=str,
        default="datasets/bigearthnet/processed/test.json",
        help="Path to test data JSON",
    )
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--compare", action="store_true", help="Compare base vs adapted")
    parser.add_argument("--adapted-only", action="store_true", help="Evaluate adapted model only")

    args = parser.parse_args()

    if args.compare:
        compare_base_vs_adapted(args.test_data, args.max_samples)
    elif args.adapted_only:
        results = evaluate_model(args.test_data, args.max_samples, use_base=False)
        print(json.dumps(results, indent=2))
    else:
        results = evaluate_model(args.test_data, args.max_samples, use_base=True)
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
