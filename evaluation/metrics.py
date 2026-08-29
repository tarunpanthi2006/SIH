import typing

def exact_match(prediction: str, ground_truth: str) -> bool:
    """Calculates Exact Match (EM) accuracy for VQA."""
    if not prediction or not ground_truth:
        return False
    return prediction.strip().lower() == ground_truth.strip().lower()

def compute_iou(box1: typing.List[float], box2: typing.List[float]) -> float:
    """
    Computes Intersection over Union (IoU) for two bounding boxes.
    Boxes should be in format [xmin, ymin, xmax, ymax] normalized [0, 1].
    """
    if len(box1) != 4 or len(box2) != 4:
        return 0.0

    # Calculate intersection coordinates
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # If the boxes do not intersect
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate areas
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Prevent division by zero
    union_area = float(box1_area + box2_area - intersection_area)
    if union_area <= 0:
        return 0.0

    iou = intersection_area / union_area
    return iou
