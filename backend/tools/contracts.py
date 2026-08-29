"""
SatQuery — Specialist Output Contract
======================================

Universal data structures for all specialist model outputs.
Every tool (change detection, optical+SAR, multispectral) returns
a `SpecialistOutput` so Person 1's agent can consume them uniformly.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EvidenceType(str, Enum):
    """Kind of spatial evidence attached to an output."""
    MASK = "mask"
    BBOX = "bbox"
    POLYGON = "polygon"
    HEATMAP = "heatmap"
    CLASSIFICATION_MAP = "classification_map"


class TaskType(str, Enum):
    """Recognised specialist task types."""
    CHANGE_DETECTION = "change_detection"
    OPTICAL_SAR = "optical_sar"
    MULTISPECTRAL = "multispectral"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class SpatialEvidence(BaseModel):
    """One piece of spatial evidence (mask, bbox, …)."""
    type: EvidenceType
    path: Optional[str] = Field(
        None, description="Path to the evidence file (PNG mask, GeoTIFF, …)"
    )
    data: Optional[Any] = Field(
        None,
        description=(
            "Inline data when a file is not appropriate "
            "(e.g. bbox coordinates, polygon vertices)"
        ),
    )
    description: Optional[str] = None
    crs: Optional[str] = Field(None, description="Coordinate reference system if georeferenced")


class Artifact(BaseModel):
    """Any file produced as a side-effect (visualisations, overlays, …)."""
    path: str
    description: Optional[str] = None
    mime_type: Optional[str] = None


# ---------------------------------------------------------------------------
# Main output
# ---------------------------------------------------------------------------

class SpecialistOutput(BaseModel):
    """
    The universal output contract for every specialist model invocation.

    Example
    -------
    >>> out = SpecialistOutput(
    ...     task=TaskType.CHANGE_DETECTION,
    ...     model="ChangeFormer",
    ...     answer="Change detected in the north-east quadrant.",
    ...     confidence=0.93,
    ...     spatial_evidence=[
    ...         SpatialEvidence(type=EvidenceType.MASK, path="outputs/mask.png")
    ...     ],
    ...     statistics={"changed_pixels": 12540, "changed_fraction": 0.048},
    ... )
    """

    # Identity
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    task: TaskType
    model: str

    # Core answer
    answer: str = ""
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    # Spatial evidence & stats
    spatial_evidence: list[SpatialEvidence] = Field(default_factory=list)
    statistics: dict[str, Any] = Field(default_factory=dict)

    # Side-effect files
    artifacts: list[Artifact] = Field(default_factory=list)

    # Free-form metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Warnings / non-fatal issues
    warnings: list[str] = Field(default_factory=list)

    # Timing
    inference_time_s: Optional[float] = None
    timestamp: float = Field(default_factory=time.time)

    # Error flag (False = success)
    is_error: bool = False
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_success(
    task: TaskType,
    model: str,
    answer: str,
    confidence: float = 0.0,
    spatial_evidence: list[SpatialEvidence] | None = None,
    statistics: dict[str, Any] | None = None,
    artifacts: list[Artifact] | None = None,
    metadata: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    inference_time_s: float | None = None,
) -> SpecialistOutput:
    """Build a successful specialist output."""
    return SpecialistOutput(
        task=task,
        model=model,
        answer=answer,
        confidence=confidence,
        spatial_evidence=spatial_evidence or [],
        statistics=statistics or {},
        artifacts=artifacts or [],
        metadata=metadata or {},
        warnings=warnings or [],
        inference_time_s=inference_time_s,
        is_error=False,
    )


def make_error(
    task: TaskType,
    model: str,
    error_message: str,
    warnings: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> SpecialistOutput:
    """Build an error specialist output (no crash, structured failure)."""
    return SpecialistOutput(
        task=task,
        model=model,
        answer="",
        confidence=0.0,
        is_error=True,
        error_message=error_message,
        warnings=warnings or [],
        metadata=metadata or {},
    )
