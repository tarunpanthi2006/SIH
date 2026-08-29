"""
SatQuery — Tool Interfaces

Defines the abstract base class that every tool (mock or real) must implement.
Person 2 and Person 3 will sub-class `BaseTool` for their real models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from backend.api.schemas import Modality, TaskType, ToolResult


class BaseTool(ABC):
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

    @abstractmethod
    async def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> ToolResult:
        """
        Run the tool on the provided inputs.

        Parameters
        ----------
        inputs : dict
            Keys vary per tool.  Common keys:
            - ``image_path``  : str — path to a single image
            - ``image_a``     : str — first image (temporal / cross-modal)
            - ``image_b``     : str — second image
            - ``question``    : str — natural-language query
            - ``query``       : str — alias for question
            - ``change_mask`` : str — path to a pre-computed change mask
        params : dict, optional
            Tool-specific hyper-parameters (thresholds, etc.).

        Returns
        -------
        ToolResult
            Standardized output matching the schema contract.
        """
        ...

    # ------------------------------------------------------------------ #
    # Convenience helpers (non-abstract)
    # ------------------------------------------------------------------ #

    def make_result(self, **kwargs: Any) -> ToolResult:
        """Create a ``ToolResult`` pre-filled with this tool's identity."""
        defaults: dict[str, Any] = {
            "task": self.task,
            "model": self.model_name,
        }
        defaults.update(kwargs)
        return ToolResult(**defaults)
