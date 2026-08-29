"""
SatQuery — Tool Interfaces

Defines the abstract base class that every tool (mock or real) must implement.
Includes the SpecialistTool adapter for ML models.
"""

from __future__ import annotations

import abc
import asyncio
import inspect
import logging
from typing import Any

from backend.api.schemas import Modality, TaskType, ToolResult
from backend.tools.contracts import SpecialistOutput

logger = logging.getLogger(__name__)


class BaseTool(abc.ABC):
    """
    Abstract base for every registered tool.

    Subclasses must set class-level attributes and implement `execute()`.
    """

    # -- Override in subclasses --
    name: str = ""
    task: TaskType = TaskType.VQA
    model_name: str = ""
    description: str = ""
    required_modalities: list[Modality] = []
    min_images: int = 1
    max_images: int = 1
    accepts_query: bool = True
    output_types: list[str] = ["answer"]

    @abc.abstractmethod
    async def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> ToolResult:
        """
        Run the tool on the provided inputs.
        """
        ...

    def make_result(self, **kwargs: Any) -> ToolResult:
        """Create a ``ToolResult`` pre-filled with this tool's identity."""
        defaults: dict[str, Any] = {
            "task": self.task,
            "model": self.model_name,
        }
        defaults.update(kwargs)
        return ToolResult(**defaults)


class SpecialistTool(BaseTool):
    """
    Contract for a specialist analysis tool.

    Subclasses must define:
        name         – unique tool identifier
        description  – one-line description for the query agent
        run()        – async execution entry-point
    """

    _loaded: bool = False

    async def load(self) -> None:
        """Load model weights / resources.  Override if needed."""
        self._loaded = True

    async def unload(self) -> None:
        """Release GPU memory.  Override if needed."""
        self._loaded = False

    async def health_check(self) -> bool:
        """Return True if the tool is ready to serve requests."""
        return self._loaded

    @abc.abstractmethod
    async def run(self, **kwargs: Any) -> SpecialistOutput:
        """Execute the specialist analysis."""
        ...

    async def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> ToolResult:
        """
        Adapter: Bridges the PipelineExecutor (inputs/params) to the 
        Specialist model (kwargs) and maps the output back to ToolResult.
        """
        kwargs = {**inputs, **(params or {})}
        
        try:
            if inspect.iscoroutinefunction(self.run):
                spec_out = await self.run(**kwargs)
            else:
                spec_out = await asyncio.to_thread(self.run, **kwargs)
                
        except Exception as e:
            logger.error(f"SpecialistTool crashed: {e}", exc_info=True)
            return ToolResult(
                task=TaskType.COMPLEX, # Fallback
                model=self.name,
                answer="",
                confidence=0.0,
                warnings=[f"Model execution crashed: {e}"]
            )

        if spec_out.is_error:
            return ToolResult(
                task=TaskType(spec_out.task.value) if spec_out.task else TaskType.COMPLEX,
                model=spec_out.model,
                answer="",
                confidence=0.0,
                warnings=[spec_out.error_message or "Unknown error"] + spec_out.warnings
            )

        # Convert SpatialEvidence schemas
        from backend.api.schemas import SpatialEvidence, SpatialEvidenceType
        converted_evidence = []
        for ev in spec_out.spatial_evidence:
            converted_evidence.append(
                SpatialEvidence(
                    type=SpatialEvidenceType(ev.type.value),
                    path=ev.path,
                    coordinates=ev.data if isinstance(ev.data, list) else None,
                    label=ev.description,
                )
            )

        return ToolResult(
            task=TaskType(spec_out.task.value),
            model=spec_out.model,
            answer=spec_out.answer,
            confidence=spec_out.confidence,
            spatial_evidence=converted_evidence,
            statistics=spec_out.statistics,
            artifacts=[a.path for a in spec_out.artifacts],
            metadata=spec_out.metadata,
            warnings=spec_out.warnings,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} loaded={self._loaded}>"
