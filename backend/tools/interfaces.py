"""
SatQuery Tool Interfaces — Common Output Contract
===================================================
All tool wrappers (VQA, Caption, Grounding) return ToolResult objects
that conform to the agreed JSON schema for Person 1's agent to consume.

This module is Person 2's contract with Person 1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class TaskType(str, Enum):
    """Supported tool task types."""
    VQA = "vqa"
    CAPTION = "caption"
    GROUNDING = "grounding"
    CHANGE_VQA = "change_vqa"


class SpatialEvidenceType(str, Enum):
    """Types of spatial evidence a tool can return."""
    BBOX = "bbox"
    MASK = "mask"
    POINT = "point"
    POLYGON = "polygon"


@dataclass
class SpatialEvidence:
    """
    A piece of spatial evidence from a tool.

    - bbox: coordinates = [x1, y1, x2, y2] normalized to [0, 1]
    - mask: path = file path to a binary mask image
    - point: coordinates = [x, y] normalized to [0, 1]
    - polygon: coordinates = [[x1,y1], [x2,y2], ...] normalized to [0, 1]
    """
    type: str
    coordinates: list[float] | None = None
    path: str | None = None
    label: str | None = None
    confidence: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        result = {"type": self.type}
        if self.coordinates is not None:
            result["coordinates"] = self.coordinates
        if self.path is not None:
            result["path"] = self.path
        if self.label is not None:
            result["label"] = self.label
        if self.confidence is not None:
            result["confidence"] = self.confidence
        return result


@dataclass
class ToolResult:
    """
    Standard output from any SatQuery tool.

    This is the common contract between Person 2 (VLM tools)
    and Person 1 (agent + tool registry).

    Every tool function (run_vqa, run_caption, run_grounding)
    MUST return a ToolResult serialized as a dict.
    """
    task: str
    model: str
    answer: str
    confidence: float = 0.0
    spatial_evidence: list[dict] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to the agreed JSON contract format."""
        return {
            "task": self.task,
            "model": self.model,
            "answer": self.answer,
            "confidence": round(self.confidence, 4),
            "spatial_evidence": self.spatial_evidence,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
            "warnings": self.warnings,
        }

    @staticmethod
    def error(task: str, model: str, error_msg: str) -> dict:
        """
        Create an error result. Used when inference fails gracefully.
        """
        return ToolResult(
            task=task,
            model=model,
            answer="",
            confidence=0.0,
            warnings=[error_msg],
            metadata={"error": True},
        ).to_dict()


def validate_image_path(image_path: str) -> tuple[bool, str]:
    """
    Validate that an image path exists and is a supported format.
    Returns (is_valid, error_message).
    """
    import os

    if not image_path:
        return False, "Image path is empty"

    if not os.path.exists(image_path):
        return False, f"Image file not found: {image_path}"

    supported_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in supported_extensions:
        return False, f"Unsupported image format: {ext}. Supported: {supported_extensions}"

    return True, ""


class Timer:
    """Simple context manager for timing tool execution."""

    def __init__(self):
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
