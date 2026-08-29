import numpy as np

def calculate_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Calculate F1, IoU, Precision, Recall."""
    # Ensure boolean
    pred = pred > 0
    target = target > 0

    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, np.logical_not(target)).sum()
    fn = np.logical_and(np.logical_not(pred), target).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(target)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    oa = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "f1": f1,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "oa": oa
    }
