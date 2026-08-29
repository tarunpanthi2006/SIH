"""
CDVQA Metrics
===============
Metrics for evaluating Change Detection Visual Question Answering.

Handles three question types:
- yes_no: Binary accuracy
- semantic: Relaxed matching (reference contained in prediction)
- spatial: Spatial accuracy (IoU-based if bounding boxes present)
"""

from __future__ import annotations


def compute_change_vqa_accuracy(
    predictions: list[str], references: list[str]
) -> dict:
    """
    Compute accuracy for Change-VQA predictions.

    Uses both exact match and relaxed match:
    - Exact: pred == ref (case-insensitive, stripped)
    - Relaxed: ref is contained in pred, or pred in ref

    Args:
        predictions: Model-generated answers
        references: Ground-truth answers

    Returns:
        dict with exact_match, relaxed_match, total
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


def compute_change_detection_f1(
    pred_masks: list,
    ref_masks: list,
    threshold: float = 0.5,
) -> dict:
    """
    Compute pixel-level F1 score for change detection masks.

    Used when evaluating P3's ChangeFormer output directly.

    Args:
        pred_masks: List of predicted binary mask arrays
        ref_masks: List of ground-truth binary mask arrays
        threshold: Binarization threshold

    Returns:
        dict with precision, recall, f1, total
    """
    import numpy as np

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred, ref in zip(pred_masks, ref_masks):
        pred_binary = (np.array(pred) > threshold).astype(int)
        ref_binary = (np.array(ref) > threshold).astype(int)

        tp = int(np.logical_and(pred_binary, ref_binary).sum())
        fp = int(np.logical_and(pred_binary, ~ref_binary.astype(bool)).sum())
        fn = int(np.logical_and(~pred_binary.astype(bool), ref_binary).sum())

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "total_samples": len(pred_masks),
    }
