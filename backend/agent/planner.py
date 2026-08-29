"""
SatQuery — Workflow Planner (LLM Powered)

Translates a (task, query, images) triple into a concrete
`ExecutionPlan` — an ordered list of tool invocations with
their inputs and parameters using Google Gemini structured output.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from backend.agent.registry import ToolRegistry
from backend.api.schemas import (
    ExecutionPlan,
    ImageMetadata,
    PlannedStep,
    TaskType,
)
from backend.config import get_settings

logger = logging.getLogger(__name__)


class LLMPlannedStep(BaseModel):
    tool: str
    inputs_json: str = Field(default="{}", description="A valid JSON string representing the inputs dictionary")
    parameters_json: str = Field(default="{}", description="A valid JSON string representing the parameters dictionary")
    depends_on: list[int] = Field(default_factory=list)

class LLMPlanResponse(BaseModel):
    steps: list[LLMPlannedStep]


class WorkflowPlanner:
    """
    Builds an `ExecutionPlan` dynamically using an LLM.

    Every planned step references a tool name that **must** exist in
    the registry — the planner validates this before returning.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def plan(
        self,
        task: TaskType,
        image_paths: list[str],
        query: str,
        metadata: list[ImageMetadata] | None = None,
        chat_history: list[dict] | None = None,
    ) -> ExecutionPlan:
        """
        Produce an execution plan for the given task using Gemini.

        Parameters
        ----------
        task : TaskType
        image_paths : list[str]
        query : str
        metadata : list[ImageMetadata], optional

        Returns
        -------
        ExecutionPlan

        Raises
        ------
        ValueError
            If any planned tool is not in the registry or API key is missing.
        """
        # ---- Ensure API Key ----
        settings = get_settings()
        if not settings.gemini_api_key or settings.gemini_api_key == "your_api_key_here":
            logger.error("GEMINI_API_KEY is not set. Cannot use LLM Planner.")
            raise ValueError("GEMINI_API_KEY is missing or invalid. Please check your .env file.")

        client = genai.Client(api_key=settings.gemini_api_key)

        # ---- Fetch available tools for the prompt ----
        available_tools = []
        for name in self.registry.tool_names:
            desc = self.registry.get_descriptor(name)
            available_tools.append(
                f"- {name}: {desc.description} (Outputs: {desc.output_types})"
            )
        tools_list_str = "\n".join(available_tools)

        # ---- Construct Prompt ----
        system_instruction = (
            "You are an expert AI orchestrator for a remote-sensing system.\n"
            f"Your job is to generate a step-by-step ExecutionPlan for the task: {task.value}.\n"
            "You have access to the following tools:\n"
            f"{tools_list_str}\n\n"
            "Rules for building the plan:\n"
            "1. You must output a list of PlannedStep objects.\n"
            "2. Each step must use a valid tool name from the list above.\n"
            "3. If a tool depends on the output of a previous tool, add the previous step's index to `depends_on`, "
            "and reference the output using `$step_N.output_type` (e.g., `$step_0.spatial_evidence`).\n"
            "4. Provide the correct inputs (like 'image_path', 'image_a', 'image_b', 'question', 'query') as required by the tools.\n"
        )

        history_str = ""
        if chat_history:
            history_str = "Chat History:\n"
            for msg in chat_history[-6:]:
                history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"
            history_str += "\n"

        user_message = (
            f"{history_str}"
            f"Task: {task.value}\n"
            f"Current Query: {query}\n"
            f"Images: {image_paths}\n\n"
            "Generate the steps for the ExecutionPlan."
        )

        # ---- Call LLM via Async API ----
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=LLMPlanResponse,
                        temperature=0.0,
                    ),
                )
                
                import json
                if response.parsed:
                    raw_steps = response.parsed.steps
                else:
                    data = json.loads(response.text)
                    raw_steps = [LLMPlannedStep(**s) for s in data.get("steps", [])]
                    
                steps = []
                for s in raw_steps:
                    try:
                        inputs = json.loads(s.inputs_json)
                    except Exception:
                        inputs = {}
                    try:
                        parameters = json.loads(s.parameters_json)
                    except Exception:
                        parameters = {}
                    steps.append(PlannedStep(
                        tool=s.tool,
                        inputs=inputs if isinstance(inputs, dict) else {},
                        parameters=parameters if isinstance(parameters, dict) else {},
                        depends_on=s.depends_on
                    ))
                break  # Exit loop on success
                    
            except Exception as e:
                if attempt == 2:
                    logger.error(f"LLM Planning failed after 3 attempts: {e}.")
                    raise RuntimeError(f"Failed to generate LLM plan: {e}")
                logger.warning(f"Gemini API error in planning (attempt {attempt+1}/3), retrying in 2s: {e}")
                await asyncio.sleep(2)

        # ---- Validate every planned tool exists ----
        for step in steps:
            if not self.registry.has(step.tool):
                raise ValueError(
                    f"Planner references tool '{step.tool}' which is not "
                    f"registered. Available: {self.registry.tool_names}"
                )

        return ExecutionPlan(task=task, steps=steps, query=query)
