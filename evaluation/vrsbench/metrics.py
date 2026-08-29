"""
VRSBench-specific Metrics
============================
Re-exports common metrics used by VRSBench evaluation.

VRSBench evaluates:
- VQA: Exact match, relaxed match accuracy
- Captioning: BLEU-4, METEOR
- Grounding: IoU@0.5

All core metric functions live in training/finetuning/evaluate.py
and evaluation/metrics.py to avoid duplication.
"""

from evaluation.metrics import exact_match, compute_iou
from training.finetuning.evaluate import (
    compute_vqa_accuracy,
    compute_caption_metrics,
    compute_grounding_iou,
)

__all__ = [
    "exact_match",
    "compute_iou",
    "compute_vqa_accuracy",
    "compute_caption_metrics",
    "compute_grounding_iou",
]
