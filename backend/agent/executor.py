"""
SatQuery — Pipeline Executor

Executes an ``ExecutionPlan`` step-by-step, resolving inter-step
references (``$step_N.field``) and collecting ``ToolResult`` objects.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from typing import Any

from backend.agent.registry import ToolRegistry
from backend.api.schemas import ExecutionPlan, ExecutionStep, ToolResult

logger = logging.getLogger(__name__)

# Pattern for inter-step references like "$step_0.spatial_evidence"
_REF_PATTERN = re.compile(r"^\$step_(\d+)\.(.+)$")


class PipelineExecutor:
    """
    Executes a plan by invoking tools from the registry in order.

    Inter-step references (``$step_N.field``) are resolved automatically.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def execute(
        self,
        plan: ExecutionPlan,
    ) -> tuple[list[ToolResult], list[ExecutionStep]]:
        """
        Execute every step in the plan.

        Parameters
        ----------
        plan : ExecutionPlan

        Returns
        -------
        tuple[list[ToolResult], list[ExecutionStep]]
            Tool results and execution trace steps (matched by index).
        """
        results: list[ToolResult] = []
        trace_steps: list[ExecutionStep] = []

        for idx, planned in enumerate(plan.steps):
            tool = self.registry.get(planned.tool)

            # Resolve inter-step references in inputs
            resolved_inputs = self._resolve_refs(planned.inputs, results)

            logger.info(
                "Executing step %d: tool='%s' model='%s'",
                idx, planned.tool, tool.model_name,
            )

            t0 = time.perf_counter()
            try:
                if inspect.iscoroutinefunction(tool.execute):
                    result = await tool.execute(resolved_inputs, planned.parameters or None)
                else:
                    # Run CPU-bound ML models in a background thread to keep FastAPI responsive
                    result = await asyncio.to_thread(tool.execute, resolved_inputs, planned.parameters or None)
                duration_ms = (time.perf_counter() - t0) * 1000
                status = "success"
                error = None
            except Exception as exc:
                duration_ms = (time.perf_counter() - t0) * 1000
                logger.error("Step %d failed: %s", idx, exc, exc_info=True)
                result = ToolResult(
                    task=plan.task,
                    model=tool.model_name,
                    answer="",
                    confidence=0.0,
                    warnings=[f"Tool execution failed: {exc}"],
                )
                status = "error"
                error = str(exc)

            results.append(result)

            # Build trace step
            outputs = self._summarize_outputs(result)
            trace_steps.append(ExecutionStep(
                step_index=idx,
                tool=planned.tool,
                model=tool.model_name,
                parameters=planned.parameters,
                outputs=outputs,
                duration_ms=round(duration_ms, 2),
                status=status,
                error=error,
            ))

        return results, trace_steps

    # ------------------------------------------------------------------ #
    # Reference resolution
    # ------------------------------------------------------------------ #

    def _resolve_refs(
        self,
        inputs: dict[str, Any],
        prior_results: list[ToolResult],
    ) -> dict[str, Any]:
        """
        Replace ``$step_N.field`` strings with actual values from
        prior results.
        """
        resolved: dict[str, Any] = {}
        for key, value in inputs.items():
            if isinstance(value, str):
                match = _REF_PATTERN.match(value)
                if match:
                    step_idx = int(match.group(1))
                    field = match.group(2)
                    if step_idx < len(prior_results):
                        ref_result = prior_results[step_idx]
                        resolved[key] = getattr(ref_result, field, value)
                        continue
            resolved[key] = value
        return resolved

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _summarize_outputs(result: ToolResult) -> list[str]:
        """List the names of non-empty outputs for the trace."""
        outputs: list[str] = []
        if result.answer:
            outputs.append("answer")
        if result.spatial_evidence:
            for ev in result.spatial_evidence:
                outputs.append(ev.type.value)
        if result.statistics:
            outputs.append("statistics")
        if result.artifacts:
            outputs.append("artifacts")
        return outputs
