"""
SatQuery — Temporal Compatibility Validation

Validates date ordering and temporal gap for bi-temporal image pairs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from backend.api.schemas import ImageMetadata, ValidationIssue

logger = logging.getLogger(__name__)

# Thresholds
_MAX_GAP_DAYS = 365 * 10  # 10 years
_MIN_GAP_DAYS = 1          # 1 day


def check_temporal_compatibility(
    meta_a: ImageMetadata,
    meta_b: ImageMetadata,
) -> list[ValidationIssue]:
    """
    Validate temporal relationship between two images.

    Checks:
    1. Both images should have acquisition dates.
    2. Date ordering (before < after).
    3. Reasonable temporal gap.

    Parameters
    ----------
    meta_a, meta_b : ImageMetadata
        Metadata for the two images (a is expected to be "before").

    Returns
    -------
    list[ValidationIssue]
    """
    issues: list[ValidationIssue] = []

    date_a = _parse_date(meta_a.acquisition_date)
    date_b = _parse_date(meta_b.acquisition_date)

    # ---- Missing dates ---- #
    if date_a is None and date_b is None:
        issues.append(ValidationIssue(
            code="MISSING_DATES",
            message="Neither image has an acquisition date. Temporal analysis may be unreliable.",
            severity="warning",
        ))
        return issues

    if date_a is None or date_b is None:
        missing = "first" if date_a is None else "second"
        issues.append(ValidationIssue(
            code="MISSING_DATE",
            message=f"The {missing} image has no acquisition date.",
            severity="warning",
        ))
        return issues

    # ---- Date ordering ---- #
    if date_a > date_b:
        issues.append(ValidationIssue(
            code="TEMPORAL_ORDER",
            message=(
                f"First image ({date_a.date()}) is newer than second "
                f"({date_b.date()}). Images may be in wrong order."
            ),
            severity="warning",
        ))

    # ---- Temporal gap ---- #
    gap = abs((date_b - date_a).days)

    if gap > _MAX_GAP_DAYS:
        issues.append(ValidationIssue(
            code="TEMPORAL_GAP_LARGE",
            message=(
                f"Temporal gap is {gap} days (~{gap // 365} years). "
                "Very large gaps may reduce change-detection accuracy."
            ),
            severity="warning",
        ))

    if gap < _MIN_GAP_DAYS:
        issues.append(ValidationIssue(
            code="TEMPORAL_GAP_SMALL",
            message=(
                f"Temporal gap is {gap} day(s). "
                "Images may represent the same scene — limited change expected."
            ),
            severity="warning",
        ))

    return issues


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y:%m:%d %H:%M:%S",
    "%Y%m%d",
    "%d/%m/%Y",
]


def _parse_date(raw: str | None) -> datetime | None:
    """Best-effort datetime parsing."""
    if not raw:
        return None
    raw = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    try:
        from dateutil.parser import parse  # type: ignore[import-untyped]
        return parse(raw)
    except Exception:
        logger.warning("Could not parse date: %s", raw)
        return None
