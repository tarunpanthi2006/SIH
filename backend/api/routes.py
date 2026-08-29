"""
SatQuery — API Routes

All ``/api/v1/*`` endpoints.  The ``/analyze`` endpoint orchestrates
the full pipeline: ingest → validate → route → plan → execute →
fuse → confidence → trace → respond.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid

from fastapi import APIRouter, HTTPException

from backend.agent.executor import PipelineExecutor
from backend.agent.planner import WorkflowPlanner
from backend.agent.registry import get_registry
from backend.agent.router import classify_task
from backend.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorDetail,
    ErrorResponse,
    ExecutionSummary,
    HealthResponse,
    TaskType,
    ToolDescriptor,
)
from backend.config import get_settings
from backend.evidence.confidence import ConfidenceEngine
from backend.evidence.fusion import EvidenceFusion
from backend.trace.execution import ExecutionTracer
from backend.validation.modality import detect_modality
from backend.ingestion.metadata import extract_metadata
from backend.validation.validator import InputValidator

logger = logging.getLogger(__name__)

# ================================================================== #
# State (Memory & Caching)
# ================================================================== #

_RESPONSE_CACHE: dict[str, AnalyzeResponse] = {}
_SESSION_MEMORY: dict[str, list[dict]] = {}


def _generate_cache_key(request: AnalyzeRequest) -> str:
    """Generate a unique hash for a request to enable exact-match caching."""
    image_paths = sorted([img.path for img in request.images])
    data = {
        "session_id": request.session_id,
        "query": request.query.strip().lower(),
        "images": image_paths,
        "hint": request.task_hint.value if request.task_hint else None
    }
    encoded = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()


router = APIRouter(prefix="/api/v1", tags=["SatQuery API v1"])


# ================================================================== #
# Health
# ================================================================== #

@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health / readiness check."""
    settings = get_settings()
    return HealthResponse(mock_mode=settings.mock_mode)


# ================================================================== #
# Tools & Tasks
# ================================================================== #

@router.get("/tools", response_model=list[ToolDescriptor])
async def list_tools() -> list[ToolDescriptor]:
    """List all registered tools."""
    return get_registry().list_tools()


@router.get("/tasks", response_model=list[str])
async def list_tasks() -> list[str]:
    """List all supported task types."""
    return [t.value for t in TaskType]


# ================================================================== #
# Analyze — Main endpoint
# ================================================================== #

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Full analysis pipeline.

    1. Extract metadata for each image.
    2. Validate inputs.
    3. Classify task (or honour ``task_hint``).
    4. Re-validate with task context.
    5. Plan workflow.
    6. Execute pipeline.
    7. Fuse evidence.
    8. Compute confidence.
    9. Build execution trace.
    10. Return structured response.
    """
    request_id = uuid.uuid4().hex
    registry = get_registry()

    # ---- 0. Caching ---- #
    cache_key = _generate_cache_key(request)
    if cache_key in _RESPONSE_CACHE:
        logger.info(f"Cache hit for request {request_id} (key: {cache_key})")
        cached_resp = _RESPONSE_CACHE[cache_key]
        return cached_resp.model_copy(update={"request_id": request_id})
        
    # ---- 0.5. Memory Fetch ---- #
    chat_history = None
    if request.session_id:
        chat_history = _SESSION_MEMORY.get(request.session_id, [])

    # ---- 1. Metadata extraction ---- #
    image_paths = [img.path for img in request.images]
    metadata_list = [extract_metadata(p) for p in image_paths]

    # Update modality from user hints if provided
    for img_input, meta in zip(request.images, metadata_list):
        if img_input.modality:
            meta.modality = img_input.modality
        else:
            meta.modality = detect_modality(meta)
        if img_input.acquisition_date:
            meta.acquisition_date = img_input.acquisition_date

    # ---- 2. Initial validation (format-level) ---- #
    validator = InputValidator()
    val_result = validator.validate(image_paths, pre_extracted_metadata=metadata_list)
    if not val_result.valid:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                errors=[
                    ErrorDetail(code=i.code, message=i.message)
                    for i in val_result.issues
                ],
                request_id=request_id,
            ).model_dump(),
        )
    # Attach extracted metadata to validation result
    val_result.metadata = metadata_list

    # ---- 3. Task classification ---- #
    modalities = [m.modality for m in metadata_list]
    task = await classify_task(
        query=request.query,
        image_count=len(image_paths),
        modalities=modalities,
        task_hint=request.task_hint,
        chat_history=chat_history,
    )

    # ---- 4. Task-aware validation ---- #
    task_val = validator.validate(image_paths, task=task, pre_extracted_metadata=metadata_list)
    # Merge warnings (task-aware may add modality / count issues)
    all_warnings = [w.message for w in val_result.warnings] + [
        w.message for w in task_val.warnings
    ]
    if not task_val.valid:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                errors=[
                    ErrorDetail(code=i.code, message=i.message)
                    for i in task_val.issues
                ],
                request_id=request_id,
            ).model_dump(),
        )

    # ---- 5. Plan workflow ---- #
    tracer = ExecutionTracer(request_id=request_id, task=task)
    planner = WorkflowPlanner(registry)
    try:
        plan = await planner.plan(
            task=task,
            image_paths=image_paths,
            query=request.query,
            metadata=metadata_list,
            chat_history=chat_history,
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ---- 6. Execute pipeline ---- #
    executor = PipelineExecutor(registry)
    tool_results, trace_steps = await executor.execute(plan)
    tracer.add_steps(trace_steps)

    # ---- 7. Fuse evidence ---- #
    fusion = EvidenceFusion()
    evidence = fusion.fuse(tool_results)

    # ---- 8. Confidence ---- #
    confidence_engine = ConfidenceEngine()
    confidence = confidence_engine.compute(
        tool_results=tool_results,
        validation=val_result,
        evidence=evidence,
    )

    # ---- 9. Execution summary ---- #
    execution = tracer.finalize()

    # ---- 10. Response ---- #
    final_response = AnalyzeResponse(
        request_id=request_id,
        task=task,
        answer=evidence.primary_answer,
        confidence=confidence,
        evidence=evidence,
        execution=execution,
        validation=val_result,
        warnings=all_warnings,
    )
    
    # ---- 11. Save State ---- #
    _RESPONSE_CACHE[cache_key] = final_response
    
    if request.session_id:
        if request.session_id not in _SESSION_MEMORY:
            _SESSION_MEMORY[request.session_id] = []
        _SESSION_MEMORY[request.session_id].append({"role": "user", "content": request.query})
        _SESSION_MEMORY[request.session_id].append({"role": "assistant", "content": final_response.answer})
        
    return final_response


# ================================================================== #
# Execution retrieval
# ================================================================== #

@router.get("/execution/{request_id}", response_model=ExecutionSummary)
async def get_execution(request_id: str) -> ExecutionSummary:
    """Retrieve a stored execution summary by request ID."""
    summary = ExecutionTracer.get_by_id(request_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Execution trace not found.")
    return summary
