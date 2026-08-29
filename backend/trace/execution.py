"""
SatQuery — Execution Trace

Builds and stores observable execution summaries.
Only exposes what the PS requires: task selected, models used,
key parameters, outputs, timing.  No chain-of-thought.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any

from backend.api.schemas import ExecutionStep, ExecutionSummary, TaskType


class ExecutionTracer:
    """
    Accumulates steps and produces an ``ExecutionSummary``.

    Also maintains an in-memory store keyed by ``request_id``
    so that ``GET /api/v1/execution/{id}`` can retrieve past traces.
    """

    # Class-level store (shared across instances)
    _store: dict[str, ExecutionSummary] = {}
    _lock = threading.Lock()

    def __init__(self, request_id: str, task: TaskType) -> None:
        self.request_id = request_id
        self.task = task
        self._steps: list[ExecutionStep] = []
        self._start = time.perf_counter()

    # ------------------------------------------------------------------ #
    # Building
    # ------------------------------------------------------------------ #

    def add_step(self, step: ExecutionStep) -> None:
        """Append an execution step."""
        self._steps.append(step)

    def add_steps(self, steps: list[ExecutionStep]) -> None:
        """Append multiple steps at once."""
        self._steps.extend(steps)

    def finalize(self) -> ExecutionSummary:
        """
        Build the summary, store it, and return it.

        Returns
        -------
        ExecutionSummary
        """
        total_ms = (time.perf_counter() - self._start) * 1000

        # Unique models used (preserving order)
        seen: set[str] = set()
        models_used: list[str] = []
        for step in self._steps:
            if step.model and step.model not in seen:
                models_used.append(step.model)
                seen.add(step.model)

        summary = ExecutionSummary(
            request_id=self.request_id,
            task=self.task,
            steps=self._steps,
            total_duration_ms=round(total_ms, 2),
            models_used=models_used,
            timestamp=datetime.now(tz=__import__('datetime').timezone.utc).isoformat(),
        )

        # Persist in in-memory store
        with self._lock:
            self._store[self.request_id] = summary

        return summary

    # ------------------------------------------------------------------ #
    # Retrieval (class methods)
    # ------------------------------------------------------------------ #

    @classmethod
    def get_by_id(cls, request_id: str) -> ExecutionSummary | None:
        """Retrieve a previously stored execution summary."""
        with cls._lock:
            return cls._store.get(request_id)

    @classmethod
    def clear_store(cls) -> None:
        """Clear the in-memory store (useful in tests)."""
        with cls._lock:
            cls._store.clear()
