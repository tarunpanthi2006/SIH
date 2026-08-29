"""
SatQuery — Input Validator (Orchestrator)

Central validation entry point.  Aggregates format, spatial, temporal,
and modality checks into a single ``ValidationResult``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from backend.api.schemas import (
    ImageMetadata,
    Modality,
    TaskType,
    ValidationIssue,
    ValidationResult,
)
from backend.ingestion.metadata import extract_metadata
from backend.validation.modality import check_modality_compatibility, detect_modality
from backend.validation.spatial import check_spatial_compatibility
from backend.validation.temporal import check_temporal_compatibility

logger = logging.getLogger(__name__)

# Supported file extensions (mirrors loader.py)
_SUPPORTED_EXTENSIONS = {
    ".tif", ".tiff", ".geotiff",
    ".png", ".jpg", ".jpeg", ".bmp", ".webp",
}

# Maximum file size — configurable via settings but kept here as a default
_DEFAULT_MAX_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB


class InputValidator:
    """
    Validates one or more images against a (possibly inferred) task.

    Usage::

        validator = InputValidator()
        result = validator.validate(image_paths, task)
    """

    def __init__(self, max_size_bytes: int = _DEFAULT_MAX_SIZE_BYTES) -> None:
        self.max_size_bytes = max_size_bytes

    def validate(
        self,
        image_paths: list[str],
        task: TaskType | None = None,
    ) -> ValidationResult:
        """
        Run all validation checks.

        Parameters
        ----------
        image_paths : list[str]
            Filesystem paths to the input images.
        task : TaskType, optional
            Task type (if already determined).  When ``None``, only
            format-level checks are performed.

        Returns
        -------
        ValidationResult
        """
        errors: list[ValidationIssue] = []
        warnings: list[ValidationIssue] = []
        metadata_list: list[ImageMetadata] = []

        # ---- Per-image checks ---- #
        for img_path in image_paths:
            file_issues = self._check_file(img_path)
            for issue in file_issues:
                if issue.severity == "warning":
                    warnings.append(issue)
                else:
                    errors.append(issue)

            meta = extract_metadata(img_path)
            meta.modality = detect_modality(meta)
            metadata_list.append(meta)

        # If any file-level errors, stop early
        if errors:
            return ValidationResult(
                valid=False,
                issues=errors,
                warnings=warnings,
                metadata=metadata_list,
            )

        # ---- Task-aware checks ---- #
        if task is not None:
            modalities = [m.modality for m in metadata_list]
            mod_issues = check_modality_compatibility(modalities, task)
            for issue in mod_issues:
                if issue.severity == "warning":
                    warnings.append(issue)
                else:
                    errors.append(issue)

        # ---- Pairwise checks (spatial / temporal) ---- #
        if len(metadata_list) == 2:
            meta_a, meta_b = metadata_list

            spatial_issues = check_spatial_compatibility(meta_a, meta_b)
            for issue in spatial_issues:
                if issue.severity == "warning":
                    warnings.append(issue)
                else:
                    errors.append(issue)

            # Temporal checks only for bi-temporal tasks
            if task in (TaskType.CHANGE_DETECTION, TaskType.CHANGE_VQA):
                temporal_issues = check_temporal_compatibility(meta_a, meta_b)
                for issue in temporal_issues:
                    if issue.severity == "warning":
                        warnings.append(issue)
                    else:
                        errors.append(issue)

        return ValidationResult(
            valid=len(errors) == 0,
            issues=errors,
            warnings=warnings,
            metadata=metadata_list,
        )

    # ------------------------------------------------------------------ #
    # File-level checks
    # ------------------------------------------------------------------ #

    def _check_file(self, path: str) -> list[ValidationIssue]:
        """Check existence, extension, and size of a single file."""
        issues: list[ValidationIssue] = []
        p = Path(path)

        if not p.exists():
            issues.append(ValidationIssue(
                code="FILE_NOT_FOUND",
                message=f"Image file does not exist: {path}",
            ))
            return issues

        if not p.is_file():
            issues.append(ValidationIssue(
                code="NOT_A_FILE",
                message=f"Path is not a file: {path}",
            ))
            return issues

        ext = p.suffix.lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            issues.append(ValidationIssue(
                code="UNSUPPORTED_FORMAT",
                message=(
                    f"Unsupported file format '{ext}'. "
                    f"Supported: {sorted(_SUPPORTED_EXTENSIONS)}."
                ),
            ))

        try:
            size = p.stat().st_size
            if size > self.max_size_bytes:
                issues.append(ValidationIssue(
                    code="FILE_TOO_LARGE",
                    message=(
                        f"File is {size / (1024**2):.1f} MB — exceeds limit "
                        f"of {self.max_size_bytes / (1024**2):.0f} MB."
                    ),
                ))
            if size == 0:
                issues.append(ValidationIssue(
                    code="EMPTY_FILE",
                    message=f"File is empty: {path}",
                ))
        except OSError as exc:
            issues.append(ValidationIssue(
                code="FILE_READ_ERROR",
                message=f"Cannot read file: {exc}",
            ))

        return issues
