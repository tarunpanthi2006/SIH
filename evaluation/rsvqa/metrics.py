"""
RSVQA-specific Metrics
=========================
Re-exports and extends common metrics for RSVQA evaluation.

RSVQA question types:
- Presence: "Is there a river in the image?" → yes/no
- Comparison: "Is the urban area larger than the forest?" → yes/no
- Rural/Urban: "Is this a rural or urban area?" → rural/urban
- Counting: "How many buildings are visible?" → number

Metrics:
- Overall Accuracy (OA): accuracy across all question types
- Average Accuracy (AA): mean of per-type accuracies
- Per-type accuracy breakdown
"""

from evaluation.metrics import exact_match
from training.finetuning.evaluate import compute_vqa_accuracy


def compute_rsvqa_metrics(
    predictions: list[str],
    references: list[str],
    question_types: list[str],
) -> dict:
    """
    Compute RSVQA-specific metrics: OA, AA, and per-type accuracy.

    Args:
        predictions: Model-generated answers
        references: Ground-truth answers
        question_types: Question type for each sample

    Returns:
        dict with overall_accuracy, average_accuracy, per_type results
    """
    # Overall accuracy
    overall = compute_vqa_accuracy(predictions, references)

    # Per-type accuracy
    type_preds: dict[str, list] = {}
    type_refs: dict[str, list] = {}
    for pred, ref, qtype in zip(predictions, references, question_types):
        type_preds.setdefault(qtype, []).append(pred)
        type_refs.setdefault(qtype, []).append(ref)

    per_type = {}
    for qtype in type_preds:
        per_type[qtype] = compute_vqa_accuracy(type_preds[qtype], type_refs[qtype])

    # Average accuracy (mean of per-type exact match)
    type_accuracies = [v["exact_match"] for v in per_type.values()]
    average_accuracy = (
        round(sum(type_accuracies) / len(type_accuracies), 2)
        if type_accuracies
        else 0.0
    )

    return {
        "overall_accuracy": overall["exact_match"],
        "average_accuracy": average_accuracy,
        "relaxed_match": overall["relaxed_match"],
        "per_type": per_type,
        "total": len(predictions),
    }


__all__ = [
    "exact_match",
    "compute_vqa_accuracy",
    "compute_rsvqa_metrics",
]
