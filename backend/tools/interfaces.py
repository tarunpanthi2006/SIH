"""
SatQuery — Specialist Tool Interface
======================================

Abstract base that every specialist tool (change, optical_sar, multispectral)
must implement.  Person 1's tool registry discovers and calls these.
"""

from __future__ import annotations

import abc
import logging
from typing import Any

from backend.tools.contracts import SpecialistOutput

logger = logging.getLogger(__name__)


class SpecialistTool(abc.ABC):
    """
    Contract for a specialist analysis tool.

    Subclasses must define:
        name         – unique tool identifier
        description  – one-line description for the query agent
        run()        – async execution entry-point
    """

    name: str = ""
    description: str = ""

    # Whether the underlying model is loaded
    _loaded: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """Load model weights / resources.  Override if needed."""
        self._loaded = True

    async def unload(self) -> None:
        """Release GPU memory.  Override if needed."""
        self._loaded = False

    async def health_check(self) -> bool:
        """Return True if the tool is ready to serve requests."""
        return self._loaded

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def run(self, **kwargs: Any) -> SpecialistOutput:
        """
        Execute the specialist analysis.

        Parameters are tool-specific (image paths, queries, metadata, …).
        Must always return a ``SpecialistOutput`` — never raise on bad input.
        """
        ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} loaded={self._loaded}>"
