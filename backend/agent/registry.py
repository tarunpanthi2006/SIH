"""
SatQuery — Tool Registry

Formal registry of available tools.  Supports programmatic registration,
lookup by name or task, and mock↔real swapping via ``MOCK_MODE``.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.api.schemas import TaskType, ToolDescriptor
from backend.tools.interfaces import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Central registry of all available tools.

    Usage::

        registry = ToolRegistry()
        registry.register(my_tool)
        tool = registry.get("vqa")
        result = await tool.execute(inputs, params)
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._descriptors: dict[str, ToolDescriptor] = {}

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool instance.

        Parameters
        ----------
        tool : BaseTool
            The tool to register.  ``tool.name`` must be unique.
        """
        name = tool.name
        if name in self._tools:
            logger.warning("Overwriting existing tool '%s' in registry", name)

        self._tools[name] = tool
        self._descriptors[name] = ToolDescriptor(
            name=name,
            task=tool.task,
            description=tool.description,
            model=tool.model_name,
            required_modalities=tool.required_modalities,
            min_images=tool.min_images,
            max_images=tool.max_images,
            accepts_query=tool.accepts_query,
            output_types=tool.output_types,
        )
        logger.info("Registered tool '%s' (model=%s)", name, tool.model_name)

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #

    def get(self, name: str) -> BaseTool:
        """
        Look up a tool by name.

        Raises
        ------
        KeyError
            If no tool with the given name is registered.
        """
        if name not in self._tools:
            raise KeyError(
                f"Tool '{name}' not found in registry. "
                f"Available: {sorted(self._tools.keys())}"
            )
        return self._tools[name]

    def get_descriptor(self, name: str) -> ToolDescriptor:
        """Look up the descriptor for a tool."""
        return self._descriptors[name]

    def list_tools(self) -> list[ToolDescriptor]:
        """Return descriptors for all registered tools."""
        return list(self._descriptors.values())

    def list_for_task(self, task: TaskType) -> list[ToolDescriptor]:
        """Return tool descriptors matching a given task type."""
        return [d for d in self._descriptors.values() if d.task == task]

    def has(self, name: str) -> bool:
        """Check whether a tool is registered."""
        return name in self._tools

    @property
    def tool_names(self) -> list[str]:
        return sorted(self._tools.keys())


# ================================================================== #
# Global singleton
# ================================================================== #

_global_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Return (and lazily create) the global tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry (useful in tests)."""
    global _global_registry
    _global_registry = None
