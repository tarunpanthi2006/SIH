"""
SatQuery — Spatial Compatibility Validation

Checks CRS compatibility, bounding-box overlap, and resolution
compatibility between image pairs.
"""

from __future__ import annotations

import logging

from backend.api.schemas import ImageMetadata, ValidationIssue

logger = logging.getLogger(__name__)

# Minimum overlap percentage to consider images spatially compatible
_MIN_OVERLAP_PCT = 10.0

# Maximum resolution ratio before warning
_MAX_RESOLUTION_RATIO = 10.0


def check_spatial_compatibility(
    meta_a: ImageMetadata,
    meta_b: ImageMetadata,
) -> list[ValidationIssue]:
    """
    Validate that two images are spatially compatible.

    Checks performed:
    1. CRS match (or transformability).
    2. Bounding-box overlap.
    3. Resolution compatibility.

    Parameters
    ----------
    meta_a, meta_b : ImageMetadata

    Returns
    -------
    list[ValidationIssue]
    """
    issues: list[ValidationIssue] = []

    # ---- Skip if no geospatial metadata ---- #
    if not meta_a.crs and not meta_b.crs:
        issues.append(ValidationIssue(
            code="NO_CRS",
            message="Neither image has a CRS — spatial compatibility cannot be verified.",
            severity="warning",
        ))
        return issues

    if not meta_a.crs or not meta_b.crs:
        issues.append(ValidationIssue(
            code="MISSING_CRS",
            message="One image lacks a CRS — spatial compatibility cannot be fully verified.",
            severity="warning",
        ))
        return issues

    # ---- CRS match ---- #
    if meta_a.crs != meta_b.crs:
        # Not necessarily an error — many CRSs can be reprojected
        try:
            from pyproj import CRS

            crs_a = CRS.from_user_input(meta_a.crs)
            crs_b = CRS.from_user_input(meta_b.crs)
            if crs_a != crs_b:
                issues.append(ValidationIssue(
                    code="CRS_MISMATCH",
                    message=(
                        f"CRS differs ({meta_a.crs} vs {meta_b.crs}). "
                        "Automatic reprojection may be required."
                    ),
                    severity="warning",
                ))
        except Exception:
            issues.append(ValidationIssue(
                code="CRS_MISMATCH",
                message=f"CRS differs ({meta_a.crs} vs {meta_b.crs}).",
                severity="warning",
            ))

    # ---- Bounding-box overlap ---- #
    if meta_a.bounds and meta_b.bounds:
        overlap = _compute_overlap_pct(meta_a.bounds, meta_b.bounds)
        if overlap < _MIN_OVERLAP_PCT:
            issues.append(ValidationIssue(
                code="LOW_SPATIAL_OVERLAP",
                message=(
                    f"Bounding-box overlap is only {overlap:.1f}% "
                    f"(minimum {_MIN_OVERLAP_PCT}%)."
                ),
            ))

    # ---- Resolution compatibility ---- #
    if meta_a.resolution and meta_b.resolution:
        ratio = _resolution_ratio(meta_a.resolution, meta_b.resolution)
        if ratio > _MAX_RESOLUTION_RATIO:
            issues.append(ValidationIssue(
                code="RESOLUTION_MISMATCH",
                message=(
                    f"Resolution ratio is {ratio:.1f}x "
                    f"(max recommended {_MAX_RESOLUTION_RATIO}x)."
                ),
                severity="warning",
            ))

    return issues


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _compute_overlap_pct(
    bounds_a: list[float], bounds_b: list[float]
) -> float:
    """
    Compute overlap percentage between two axis-aligned bounding boxes.

    Each bounds list is ``[left, bottom, right, top]``.
    Returns overlap as a percentage of the *smaller* box's area.
    """
    left = max(bounds_a[0], bounds_b[0])
    bottom = max(bounds_a[1], bounds_b[1])
    right = min(bounds_a[2], bounds_b[2])
    top = min(bounds_a[3], bounds_b[3])

    if left >= right or bottom >= top:
        return 0.0

    overlap_area = (right - left) * (top - bottom)

    area_a = (bounds_a[2] - bounds_a[0]) * (bounds_a[3] - bounds_a[1])
    area_b = (bounds_b[2] - bounds_b[0]) * (bounds_b[3] - bounds_b[1])

    if area_a == 0 and area_b == 0:
        return 0.0

    smaller = min(area_a, area_b) if min(area_a, area_b) > 0 else max(area_a, area_b)
    if smaller == 0:
        return 0.0

    return (overlap_area / smaller) * 100.0


def _resolution_ratio(
    res_a: tuple[float, float], res_b: tuple[float, float]
) -> float:
    """Return the maximum per-axis resolution ratio between two images."""
    ratios = []
    for a, b in zip(res_a, res_b):
        if a > 0 and b > 0:
            ratios.append(max(a / b, b / a))
    return max(ratios) if ratios else 1.0
