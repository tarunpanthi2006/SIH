"""
SatQuery-RS — Advanced Model Evaluation & Comparison
=======================================================
Evaluates GeoChat-7B (base) vs SatQuery-RS (adapted at checkpoint-1250)
on held-out validation data, producing a detailed comparison report.

Metrics:
  ✦ VQA: Exact Match, Relaxed Match, F1 Score
  ✦ Captioning: BLEU-4, METEOR, ROUGE-L
  ✦ Perplexity: Per-sample loss comparison
  ✦ Speed: Tokens/sec throughput benchmark

Usage:
    # Quick eval of adapted model only
    python -m training.finetuning.evaluate \\
        --test-data datasets/bigearthnet/processed/val_small.json \\
        --adapted-only --max-samples 100

    # Full comparison: base vs adapted
    python -m training.finetuning.evaluate \\
        --test-data datasets/bigearthnet/processed/val_small.json \\
        --compare --max-samples 100

    # Eval with perplexity (loss-based, no generation needed)
    python -m training.finetuning.evaluate \\
        --test-data datasets/bigearthnet/processed/val_small.json \\
        --perplexity --max-samples 200
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Metric Computation Functions
# ============================================================

def compute_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 score between prediction and reference."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * (precision * recall) / (precision + recall)


def compute_vqa_accuracy(predictions: list[str], references: list[str]) -> dict:
    """
    Compute VQA accuracy metrics.
    - Exact match: prediction == reference (case-insensitive)
    - Relaxed match: reference is contained in prediction (case-insensitive)
    - Token F1: Average token-level F1 score
    """
    exact = 0
    relaxed = 0
    f1_scores = []
    total = len(predictions)

    for pred, ref in zip(predictions, references):
        pred_clean = pred.strip().lower()
        ref_clean = ref.strip().lower()

        if pred_clean == ref_clean:
            exact += 1
            relaxed += 1
        elif ref_clean in pred_clean or pred_clean in ref_clean:
            relaxed += 1

        f1_scores.append(compute_f1(pred_clean, ref_clean))

    return {
        "exact_match": round(exact / max(total, 1) * 100, 2),
        "relaxed_match": round(relaxed / max(total, 1) * 100, 2),
        "token_f1": round(sum(f1_scores) / max(len(f1_scores), 1) * 100, 2),
        "total": total,
    }


def compute_caption_metrics(predictions: list[str], references: list[str]) -> dict:
    """
    Compute captioning metrics: BLEU-4, METEOR, ROUGE-L.
    Falls back gracefully if dependencies are unavailable.
    """
    metrics = {}

    # BLEU-4
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk

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

    # METEOR
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

    # ROUGE-L (simple implementation)
    rouge_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        if pred_tokens and ref_tokens:
            lcs_len = _lcs_length(pred_tokens, ref_tokens)
            precision = lcs_len / len(pred_tokens) if pred_tokens else 0
            recall = lcs_len / len(ref_tokens) if ref_tokens else 0
            if precision + recall > 0:
                rouge_l = 2 * precision * recall / (precision + recall)
            else:
                rouge_l = 0.0
            rouge_scores.append(rouge_l)

    if rouge_scores:
        metrics["rouge_l"] = round(sum(rouge_scores) / len(rouge_scores) * 100, 2)

    metrics["total"] = len(predictions)
    return metrics


def _lcs_length(x: list, y: list) -> int:
    """Compute length of Longest Common Subsequence."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_perplexity_comparison(
    test_data: list[dict],
    max_samples: int = 200,
) -> dict:
    """
    Compute per-sample perplexity for both base and adapted models.

    ✅ NO IMAGES REQUIRED — this is a pure text-based loss comparison.
    The model reads the question+answer text and measures how "surprised"
    it is by the correct answer. Lower loss = model learned better.

    Returns comparison of average cross-entropy loss and perplexity.
    """
    from models.vqa.model import SatQueryVLM, get_model

    results = {}

    for model_type in ["adapted", "base"]:
        logger.info(f"\n  Computing perplexity for {model_type} model...")

        # Reload model cleanly
        if SatQueryVLM._instance:
            SatQueryVLM._instance.unload()
            SatQueryVLM._instance = None

        if model_type == "base":
            os.environ["LORA_ADAPTER_PATH"] = "__none__"
        else:
            os.environ.pop("LORA_ADAPTER_PATH", None)
            # Reload .env so the correct adapter path is picked up
            from dotenv import load_dotenv
            load_dotenv(override=True)

        vlm = get_model()
        model = vlm.model
        tokenizer = vlm.tokenizer

        losses = []
        skipped = 0

        for i, sample in enumerate(test_data[:max_samples]):
            conversations = sample.get("conversations", [])
            if len(conversations) < 2:
                skipped += 1
                continue

            human_turn = conversations[0].get("value", "").replace("<image>\n", "").strip()
            gpt_turn = conversations[1].get("value", "").strip()

            if not human_turn or not gpt_turn:
                skipped += 1
                continue

            # Pure text prompt — no images needed!
            prompt = f"USER: {human_turn}\nASSISTANT: {gpt_turn}</s>"

            encoded = tokenizer(
                prompt,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"]
            if torch.cuda.is_available():
                input_ids = input_ids.to("cuda")

            try:
                with torch.inference_mode():
                    # Pass labels=input_ids so the model computes cross-entropy loss
                    # No images= kwarg — text-only forward pass
                    outputs = model(input_ids=input_ids, labels=input_ids)
                    if outputs.loss is not None:
                        loss = outputs.loss.item()
                        if not math.isnan(loss) and not math.isinf(loss):
                            losses.append(loss)
            except Exception as e:
                logger.debug(f"Perplexity sample failed: {e}")
                skipped += 1
                continue

            if (i + 1) % 50 == 0:
                avg_so_far = sum(losses) / max(len(losses), 1)
                logger.info(
                    f"    [{model_type}] {i+1}/{min(max_samples, len(test_data))} "
                    f"avg_loss={avg_so_far:.4f} (skipped={skipped})"
                )

        if not losses:
            logger.warning(f"  ⚠️  No valid samples for {model_type} — all skipped!")
            results[model_type] = {"avg_loss": None, "perplexity": None, "num_samples": 0}
            continue

        avg_loss = sum(losses) / len(losses)
        perplexity = math.exp(min(avg_loss, 100))  # Cap to prevent overflow

        results[model_type] = {
            "avg_loss": round(avg_loss, 4),
            "perplexity": round(perplexity, 2),
            "num_samples": len(losses),
            "skipped": skipped,
        }
        logger.info(
            f"  ✅ {model_type}: avg_loss={avg_loss:.4f}, "
            f"perplexity={perplexity:.2f} ({len(losses)} samples)"
        )

    # Compute improvement
    if ("base" in results and "adapted" in results
            and results["base"]["avg_loss"] is not None
            and results["adapted"]["avg_loss"] is not None):
        loss_improvement = results["base"]["avg_loss"] - results["adapted"]["avg_loss"]
        ppl_improvement = results["base"]["perplexity"] - results["adapted"]["perplexity"]
        results["improvement"] = {
            "loss_reduction": round(loss_improvement, 4),
            "perplexity_reduction": round(ppl_improvement, 2),
            "loss_reduction_pct": round(
                loss_improvement / max(results["base"]["avg_loss"], 0.001) * 100, 2
            ),
        }

    return results



# ============================================================
# Main Evaluation Functions
# ============================================================

def evaluate_model(
    test_data_path: str,
    max_samples: int = 200,
    use_base: bool = False,
) -> dict:
    """
    Evaluate a model on held-out test data using generation.

    Args:
        test_data_path: Path to test JSON file
        max_samples: Maximum samples to evaluate
        use_base: If True, evaluate base model (no LoRA adapter)
    """
    from models.vqa.inference import vqa_inference, caption_inference

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

    for sample in test_data:
        task_type = sample.get("metadata", {}).get("task_type", "unknown")
        if task_type in ("binary", "mcq"):
            vqa_samples.append(sample)
        elif task_type == "captioning":
            caption_samples.append(sample)
        else:
            # Default: treat as VQA if conversations exist
            if "conversations" in sample and len(sample["conversations"]) >= 2:
                vqa_samples.append(sample)

    results = {"model": model_label, "total_samples": len(test_data)}

    # VQA evaluation
    if vqa_samples:
        logger.info(f"  Evaluating VQA on {len(vqa_samples)} samples...")
        vqa_preds = []
        vqa_refs = []
        vqa_questions = []
        start_time = time.time()

        for i, sample in enumerate(vqa_samples):
            image_path = sample.get("image", "")
            question = sample["conversations"][0]["value"].replace("<image>\n", "")
            reference = sample["conversations"][1]["value"]

            logger.info(f"    Checking image path: {image_path} (Exists: {os.path.exists(image_path)})")
            if os.path.exists(image_path):
                try:
                    pred, _ = vqa_inference(image_path, question)
                    vqa_preds.append(pred)
                    vqa_refs.append(reference)
                    vqa_questions.append(question)
                except Exception as e:
                    logger.error(f"VQA failed: {e}", exc_info=True)

            if (i + 1) % 25 == 0:
                elapsed = time.time() - start_time
                logger.info(f"    [{model_label}] VQA: {i+1}/{len(vqa_samples)} "
                           f"({elapsed:.1f}s)")

        if vqa_preds:
            results["vqa"] = compute_vqa_accuracy(vqa_preds, vqa_refs)
            # Add per-sample examples
            results["vqa"]["examples"] = [
                {"question": q, "predicted": p, "reference": r}
                for q, p, r in zip(
                    vqa_questions[:5],
                    vqa_preds[:5],
                    vqa_refs[:5],
                )
            ]
            logger.info(f"  VQA results: {results['vqa']}")

    # Caption evaluation
    if caption_samples:
        logger.info(f"  Evaluating captioning on {len(caption_samples)} samples...")
        cap_preds = []
        cap_refs = []
        start_time = time.time()

        for i, sample in enumerate(caption_samples):
            image_path = sample.get("image", "")
            reference = sample["conversations"][1]["value"]

            logger.info(f"    Checking image path: {image_path} (Exists: {os.path.exists(image_path)})")
            if os.path.exists(image_path):
                try:
                    pred, _ = caption_inference(image_path)
                    cap_preds.append(pred)
                    cap_refs.append(reference)
                except Exception as e:
                    logger.error(f"Caption failed: {e}", exc_info=True)

            if (i + 1) % 25 == 0:
                elapsed = time.time() - start_time
                logger.info(f"    [{model_label}] Caption: {i+1}/{len(caption_samples)} "
                           f"({elapsed:.1f}s)")

        if cap_preds:
            results["caption"] = compute_caption_metrics(cap_preds, cap_refs)
            logger.info(f"  Caption results: {results['caption']}")

    return results


def compare_base_vs_adapted(
    test_data_path: str,
    max_samples: int = 200,
    include_perplexity: bool = False,
) -> dict:
    """
    Run evaluation for both base and adapted models, then compare.
    Produces a detailed comparison table.
    """
    from models.vqa.model import SatQueryVLM

    logger.info(
        f"\n{'═' * 66}\n"
        f"  🔬 BASE vs ADAPTED MODEL COMPARISON\n"
        f"  ├─ Test data: {test_data_path}\n"
        f"  ├─ Max samples: {max_samples}\n"
        f"{'═' * 66}"
    )

    # Perplexity comparison (fast, loss-based)
    perplexity_results = None
    if include_perplexity:
        with open(test_data_path, "r") as f:
            test_data = json.load(f)
        perplexity_results = compute_perplexity_comparison(test_data, max_samples)

    # --- Evaluate BASE model ---
    if SatQueryVLM._instance:
        SatQueryVLM._instance.unload()
        SatQueryVLM._instance = None

    os.environ["LORA_ADAPTER_PATH"] = "__none__"
    logger.info("\n  ─── Evaluating BASE model (GeoChat-7B) ───")
    base_results = evaluate_model(test_data_path, max_samples, use_base=True)

    # --- Evaluate ADAPTED model ---
    if SatQueryVLM._instance:
        SatQueryVLM._instance.unload()
        SatQueryVLM._instance = None

    os.environ.pop("LORA_ADAPTER_PATH", None)
    from dotenv import load_dotenv
    load_dotenv(override=True)

    logger.info("\n  ─── Evaluating ADAPTED model (SatQuery-RS) ───")
    adapted_results = evaluate_model(test_data_path, max_samples, use_base=False)

    comparison = {
        "base": base_results,
        "adapted": adapted_results,
        "perplexity": perplexity_results,
    }

    # ── Print Beautiful Comparison Table ──
    logger.info(
        f"\n{'═' * 66}\n"
        f"  📊 COMPARISON RESULTS\n"
        f"{'═' * 66}\n"
        f"  {'Metric':<30} {'Base':>12} {'Adapted':>12} {'Delta':>12}\n"
        f"  {'─' * 62}"
    )

    for task in ["vqa", "caption"]:
        if task in base_results and task in adapted_results:
            logger.info(f"  [{task.upper()}]")
            for metric, base_val in base_results[task].items():
                if metric in ("total", "examples"):
                    continue
                adapted_val = adapted_results[task].get(metric, 0)
                if isinstance(base_val, (int, float)) and isinstance(adapted_val, (int, float)):
                    delta = adapted_val - base_val
                    sign = "+" if delta >= 0 else ""
                    emoji = "🟢" if delta > 0 else ("🔴" if delta < 0 else "⚪")
                    logger.info(
                        f"  {emoji} {metric:<28} {base_val:>12.2f} {adapted_val:>12.2f} {sign}{delta:>11.2f}"
                    )

    if perplexity_results:
        logger.info(f"\n  [PERPLEXITY]")
        base_ppl = perplexity_results.get("base", {})
        adapted_ppl = perplexity_results.get("adapted", {})
        improvement = perplexity_results.get("improvement", {})

        logger.info(f"  {'avg_loss':<28} {base_ppl.get('avg_loss', 0):>12.4f} "
                    f"{adapted_ppl.get('avg_loss', 0):>12.4f} "
                    f"{improvement.get('loss_reduction', 0):>+11.4f}")
        logger.info(f"  {'perplexity':<28} {base_ppl.get('perplexity', 0):>12.2f} "
                    f"{adapted_ppl.get('perplexity', 0):>12.2f} "
                    f"{improvement.get('perplexity_reduction', 0):>+11.2f}")

    logger.info(f"  {'─' * 62}")
    logger.info(f"{'═' * 66}")

    # Save comparison
    output_path = Path("evaluation/comparison_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info(f"  Results saved to: {output_path}")

    return comparison


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SatQuery-RS — Advanced Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick adapted-only eval
  python -m training.finetuning.evaluate --adapted-only --max-samples 50

  # Full base vs adapted comparison
  python -m training.finetuning.evaluate --compare --max-samples 100

  # Perplexity comparison (fast, no generation)
  python -m training.finetuning.evaluate --perplexity --max-samples 200
        """,
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="datasets/bigearthnet/processed/val_small.json",
        help="Path to test data JSON",
    )
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--compare", action="store_true", help="Compare base vs adapted")
    parser.add_argument("--adapted-only", action="store_true", help="Evaluate adapted model only")
    parser.add_argument("--perplexity", action="store_true", help="Include perplexity comparison")

    args = parser.parse_args()

    if args.compare:
        compare_base_vs_adapted(
            args.test_data,
            args.max_samples,
            include_perplexity=args.perplexity,
        )
    elif args.perplexity:
        # Perplexity-only mode (fast)
        with open(args.test_data, "r") as f:
            test_data = json.load(f)
        results = compute_perplexity_comparison(test_data, args.max_samples)
        print(json.dumps(results, indent=2))
    elif args.adapted_only:
        results = evaluate_model(args.test_data, args.max_samples, use_base=False)
        print(json.dumps(results, indent=2, default=str))
    else:
        results = evaluate_model(args.test_data, args.max_samples, use_base=True)
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
