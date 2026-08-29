"""
SatQuery — API Schemas

Every request/response in the system is a Pydantic v2 model defined here.
This is the single source of truth for the API contract.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ================================================================== #
# Enums
# ================================================================== #

class TaskType(str, enum.Enum):
    """Supported remote-sensing tasks."""

    VQA = "vqa"
    CAPTION = "caption"
    GROUNDING = "grounding"
    CHANGE_DETECTION = "change_detection"
    CHANGE_VQA = "change_vqa"
    OPTICAL_SAR = "optical_sar"
    MULTISPECTRAL = "multispectral"
    COMPLEX = "complex"


class Modality(str, enum.Enum):
    """Image modality."""

    OPTICAL = "optical"
    SAR = "sar"
    MULTISPECTRAL = "multispectral"
    UNKNOWN = "unknown"


class SpatialEvidenceType(str, enum.Enum):
    """Type of spatial evidence returned by a tool."""

    BBOX = "bbox"
    POLYGON = "polygon"
    MASK = "mask"
    POINT = "point"


# ================================================================== #
# Image & Metadata
# ================================================================== #

class ImageInput(BaseModel):
    """A single image reference in an analysis request."""

    path: str = Field(..., description="Filesystem path to the image.")
    modality: Modality | None = Field(
        None, description="If known, the image modality. Auto-detected if omitted."
    )
    role: str | None = Field(
        None,
        description="Semantic role, e.g. 'before', 'after', 'optical', 'sar'.",
    )
    acquisition_date: str | None = Field(
        None, description="Acquisition date (ISO-8601) if known."
    )


class ImageMetadata(BaseModel):
    """Extracted metadata for one image."""

    path: str
    width: int = 0
    height: int = 0
    bands: int = 0
    dtype: str = ""
    crs: str | None = None
    bounds: list[float] | None = Field(
        None, description="[left, bottom, right, top] in CRS units."
    )
    resolution: tuple[float, float] | None = Field(
        None, description="(x_res, y_res) in CRS units."
    )
    acquisition_date: str | None = None
    sensor: str | None = None
    modality: Modality = Modality.UNKNOWN
    extra: dict[str, Any] = Field(default_factory=dict)


# ================================================================== #
# Validation
# ================================================================== #

class ValidationIssue(BaseModel):
    """A single validation error or warning."""

    code: str = Field(..., description="Machine-readable error code.")
    message: str = Field(..., description="Human-readable explanation.")
    severity: str = Field(
        "error", description="'error' or 'warning'."
    )


class ValidationResult(BaseModel):
    """Aggregated validation output."""

    valid: bool = True
    issues: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    metadata: list[ImageMetadata] = Field(default_factory=list)


# ================================================================== #
# Tool Result & Evidence
# ================================================================== #

class SpatialEvidence(BaseModel):
    """One piece of spatial evidence (bbox, mask path, polygon, etc.)."""

    type: SpatialEvidenceType
    coordinates: list[float] | None = Field(
        None,
        description="Flat coordinate list, interpretation depends on 'type'.",
    )
    path: str | None = Field(
        None, description="Path to a mask or artifact file."
    )
    label: str | None = None
    confidence: float | None = None


class ToolResult(BaseModel):
    """Standardized output from any tool in the registry."""

    task: TaskType
    model: str = ""
    answer: str = ""
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Model-level confidence."
    )
    spatial_evidence: list[SpatialEvidence] = Field(default_factory=list)
    artifacts: list[str] = Field(
        default_factory=list,
        description="Paths to generated artifacts (masks, images, etc.).",
    )
    statistics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class EvidenceBundle(BaseModel):
    """Fused evidence from one or more tool results."""

    primary_answer: str = ""
    tool_results: list[ToolResult] = Field(default_factory=list)
    spatial_evidence: list[SpatialEvidence] = Field(default_factory=list)
    statistics: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    provenance: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-tool provenance (which model said what).",
    )


# ================================================================== #
# Execution Trace
# ================================================================== #

class ExecutionStep(BaseModel):
    """One observable step in the execution pipeline."""

    step_index: int
    tool: str
    model: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    outputs: list[str] = Field(
        default_factory=list,
        description="Names of outputs produced (e.g. 'answer', 'change_mask').",
    )
    duration_ms: float = 0.0
    status: str = "success"
    error: str | None = None


class ExecutionSummary(BaseModel):
    """Observable execution trace for the full request."""

    request_id: str
    task: TaskType
    steps: list[ExecutionStep] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    models_used: list[str] = Field(default_factory=list)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(tz=__import__('datetime').timezone.utc).isoformat()
    )


# ================================================================== #
# Tool Descriptor (Registry)
# ================================================================== #

class ToolDescriptor(BaseModel):
    """Describes a registered tool's capabilities."""

    name: str
    task: TaskType
    description: str = ""
    model: str = ""
    required_modalities: list[Modality] = Field(default_factory=list)
    min_images: int = 1
    max_images: int = 1
    accepts_query: bool = True
    output_types: list[str] = Field(
        default_factory=list,
        description="Types of outputs (e.g. 'answer', 'mask', 'bbox').",
    )


# ================================================================== #
# API Request / Response
# ================================================================== #

class AnalyzeRequest(BaseModel):
    """Top-level request to POST /api/v1/analyze."""

    query: str = Field(..., min_length=1, description="Natural-language question.")
    images: list[ImageInput] = Field(
        ..., min_length=1, description="One or more image references."
    )
    task_hint: TaskType | None = Field(
        None,
        description="Optional: override automatic task routing.",
    )
    session_id: str | None = Field(
        None,
        description="Optional: unique session ID for context memory and caching.",
    )


class AnalyzeResponse(BaseModel):
    """Top-level response from POST /api/v1/analyze."""

    request_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex
    )
    task: TaskType
    answer: str
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    evidence: EvidenceBundle = Field(default_factory=EvidenceBundle)
    execution: ExecutionSummary | None = None
    validation: ValidationResult | None = None
    warnings: list[str] = Field(default_factory=list)


# ================================================================== #
# Utility Responses
# ================================================================== #

class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "0.1.0"
    mock_mode: bool = True
    timestamp: str = Field(
        default_factory=lambda: datetime.now(tz=__import__('datetime').timezone.utc).isoformat()
    )


class ErrorDetail(BaseModel):
    """A single error entry."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Structured error response."""

    errors: list[ErrorDetail]
    request_id: str | None = None


# ================================================================== #
# Workflow Planning
# ================================================================== #

class PlannedStep(BaseModel):
    """One step in an execution plan."""

    tool: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[int] = Field(
        default_factory=list,
        description="Indices of prior steps this depends on.",
    )


class ExecutionPlan(BaseModel):
    """A complete workflow plan produced by the planner."""

    task: TaskType
    steps: list[PlannedStep]
    query: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
