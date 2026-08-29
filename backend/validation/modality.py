"""
SatQuery — Modality Detection & Validation

Detects whether an image is optical, SAR, or multispectral and validates
modality compatibility against the selected task.
"""

from __future__ import annotations

from backend.api.schemas import ImageMetadata, Modality, TaskType, ValidationIssue


# ================================================================== #
# Detection
# ================================================================== #

def detect_modality(meta: ImageMetadata) -> Modality:
    """
    Determine modality from metadata.

    Priority order:
    1. Explicit ``meta.modality`` if already set and not UNKNOWN.
    2. Heuristic from band count / dtype.

    Parameters
    ----------
    meta : ImageMetadata

    Returns
    -------
    Modality
    """
    if meta.modality and meta.modality != Modality.UNKNOWN:
        return meta.modality

    # Heuristic rules
    if meta.bands == 1 and "float" in meta.dtype.lower():
        return Modality.SAR
    if meta.bands > 4:
        return Modality.MULTISPECTRAL
    if meta.bands <= 4:
        return Modality.OPTICAL

    return Modality.UNKNOWN


# ================================================================== #
# Compatibility checks
# ================================================================== #

# Required modality sets per task
_TASK_MODALITY_REQUIREMENTS: dict[TaskType, dict] = {
    TaskType.VQA: {
        "allowed": {Modality.OPTICAL, Modality.MULTISPECTRAL, Modality.SAR, Modality.UNKNOWN},
        "min_images": 1,
        "max_images": 1,
    },
    TaskType.CAPTION: {
        "allowed": {Modality.OPTICAL, Modality.MULTISPECTRAL, Modality.UNKNOWN},
        "min_images": 1,
        "max_images": 1,
    },
    TaskType.GROUNDING: {
        "allowed": {Modality.OPTICAL, Modality.MULTISPECTRAL, Modality.UNKNOWN},
        "min_images": 1,
        "max_images": 1,
    },
    TaskType.CHANGE_DETECTION: {
        "allowed": {Modality.OPTICAL, Modality.MULTISPECTRAL, Modality.SAR, Modality.UNKNOWN},
        "min_images": 2,
        "max_images": 2,
    },
    TaskType.CHANGE_VQA: {
        "allowed": {Modality.OPTICAL, Modality.MULTISPECTRAL, Modality.SAR, Modality.UNKNOWN},
        "min_images": 2,
        "max_images": 2,
    },
    TaskType.OPTICAL_SAR: {
        "allowed": {Modality.OPTICAL, Modality.SAR, Modality.UNKNOWN},
        "min_images": 2,
        "max_images": 2,
        "require_pair": (Modality.OPTICAL, Modality.SAR),
    },
    TaskType.MULTISPECTRAL: {
        "allowed": {Modality.MULTISPECTRAL, Modality.UNKNOWN},
        "min_images": 1,
        "max_images": 1,
    },
}


def check_modality_compatibility(
    modalities: list[Modality],
    task: TaskType,
) -> list[ValidationIssue]:
    """
    Validate that the detected modalities are compatible with the task.

    Parameters
    ----------
    modalities : list[Modality]
        Detected modality per image.
    task : TaskType

    Returns
    -------
    list[ValidationIssue]
    """
    issues: list[ValidationIssue] = []
    reqs = _TASK_MODALITY_REQUIREMENTS.get(task)

    if reqs is None:
        return issues

    # -- Image count --
    n = len(modalities)
    if n < reqs["min_images"]:
        issues.append(ValidationIssue(
            code="INSUFFICIENT_IMAGES",
            message=(
                f"Task '{task.value}' requires at least {reqs['min_images']} "
                f"image(s), but {n} provided."
            ),
        ))
    if n > reqs["max_images"]:
        issues.append(ValidationIssue(
            code="TOO_MANY_IMAGES",
            message=(
                f"Task '{task.value}' accepts at most {reqs['max_images']} "
                f"image(s), but {n} provided."
            ),
        ))

    # -- Modality allowed --
    allowed = reqs["allowed"]
    for i, mod in enumerate(modalities):
        if mod not in allowed:
            issues.append(ValidationIssue(
                code="MODALITY_MISMATCH",
                message=(
                    f"Image {i} has modality '{mod.value}' which is not "
                    f"supported for task '{task.value}'."
                ),
            ))

    # -- Specific pair requirement (optical_sar) --
    if "require_pair" in reqs and n == 2:
        req_a, req_b = reqs["require_pair"]
        has_a = any(m == req_a for m in modalities)
        has_b = any(m == req_b for m in modalities)
        # Allow UNKNOWN to satisfy either side
        unknowns = sum(1 for m in modalities if m == Modality.UNKNOWN)
        if not has_a and not has_b and unknowns < 2:
            issues.append(ValidationIssue(
                code="MISSING_MODALITY_PAIR",
                message=(
                    f"Task '{task.value}' requires one '{req_a.value}' and "
                    f"one '{req_b.value}' image."
                ),
            ))
        elif not has_a and unknowns == 0:
            issues.append(ValidationIssue(
                code="MISSING_MODALITY_PAIR",
                message=f"No '{req_a.value}' image found for task '{task.value}'.",
            ))
        elif not has_b and unknowns == 0:
            issues.append(ValidationIssue(
                code="MISSING_MODALITY_PAIR",
                message=f"No '{req_b.value}' image found for task '{task.value}'.",
            ))

    return issues
