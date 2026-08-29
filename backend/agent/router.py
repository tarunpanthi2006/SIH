"""
SatQuery — Agentic Task Router (LLM Powered)

Uses Google Gemini to classify a user query + image metadata into a `TaskType`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Sequence

from pydantic import BaseModel
from google import genai
from google.genai import types

from backend.api.schemas import Modality, TaskType
from backend.config import get_settings

logger = logging.getLogger(__name__)

class TaskClassification(BaseModel):
    task: TaskType
    reasoning: str


async def classify_task(
    query: str,
    image_count: int = 1,
    modalities: Sequence[Modality] = (),
    task_hint: TaskType | None = None,
    chat_history: list[dict] | None = None,
) -> TaskType:
    """
    Determine the `TaskType` for a given user query using an LLM.

    Parameters
    ----------
    query : str
        Natural-language query from the user.
    image_count : int
        Number of images provided.
    modalities : Sequence[Modality]
        Detected modality per image.
    task_hint : TaskType, optional
        If the user explicitly requests a specific task, honour it.

    Returns
    -------
    TaskType
    """
    # ---- Explicit override ----
    if task_hint is not None:
        return task_hint

    # ---- Ensure API Key ----
    settings = get_settings()

    # MOCK MODE FALLBACK
    if settings.mock_mode:
        q = query.lower()
        if "change" in q:
            return TaskType.CHANGE_DETECTION
        elif "highlight" in q or "where" in q:
            return TaskType.GROUNDING
        elif "ndvi" in q or "multispectral" in q:
            return TaskType.MULTISPECTRAL
        elif "sar" in q or "sensors" in q:
            return TaskType.OPTICAL_SAR
        else:
            return TaskType.VQA

    if not settings.gemini_api_key or settings.gemini_api_key == "your_api_key_here":
        logger.error("GEMINI_API_KEY is not set. Cannot use LLM Router.")
        raise ValueError("GEMINI_API_KEY is missing or invalid. Please check your .env file.")

    # ---- Initialize Gemini Client ----
    client = genai.Client(api_key=settings.gemini_api_key)

    # ---- Construct Prompt ----
    system_instruction = (
        "You are the routing brain for SatQuery, a remote-sensing intelligence system. "
        "Your job is to analyze the user's query and the metadata of the provided images, "
        "and select the single most appropriate TaskType to handle the request.\n\n"
        "Here are the available TaskTypes and their constraints:\n"
        "- VQA: General questions about a single image.\n"
        "- CAPTION: Describe or summarize a single image without a specific question.\n"
        "- GROUNDING: Locate, find, or highlight objects (returns bounding boxes).\n"
        "- CHANGE_DETECTION: Generate a change mask/map between two images (no question asked).\n"
        "- CHANGE_VQA: Answer a specific question about the change between two images.\n"
        "- OPTICAL_SAR: Cross-modal analysis between exactly one optical and one SAR image.\n"
        "- MULTISPECTRAL: Analysis requiring spectral indices (NDVI, etc.).\n"
        "- COMPLEX: Multi-step reasoning that doesn't fit the above.\n\n"
        "Rules:\n"
        "1. If image_count is 2 and the query asks about differences/changes, choose CHANGE_VQA.\n"
        "2. If image_count is 2 and the query just says 'detect change', choose CHANGE_DETECTION.\n"
        "3. If modalities contain exactly ['optical', 'sar'], prioritize OPTICAL_SAR unless it's a change question.\n"
        "Provide your reasoning briefly, then output the strict enum value."
    )

    history_str = ""
    if chat_history:
        history_str = "Chat History:\n"
        for msg in chat_history[-6:]:  # Keep last 3 turns
            history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"
        history_str += "\n"

    user_message = (
        f"{history_str}"
        f"Current Query: \"{query}\"\n"
        f"Image Count: {image_count}\n"
        f"Modalities: {[m.value for m in modalities]}\n\n"
        "Classify this request."
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
                    response_schema=TaskClassification,
                    temperature=0.0,
                ),
            )
            
            if response.parsed:
                result = response.parsed
                logger.info(f"LLM routed task to {result.task.value} (Reason: {result.reasoning})")
                return result.task
            else:
                import json
                data = json.loads(response.text)
                logger.info(f"LLM routed task to {data['task']}")
                return TaskType(data["task"])
                
        except Exception as e:
            if attempt == 2:
                logger.error(f"LLM Routing failed after 3 attempts: {e}. Falling back to default heuristics.")
                if image_count == 2:
                    return TaskType.CHANGE_VQA
                return TaskType.VQA
            logger.warning(f"Gemini API error in routing (attempt {attempt+1}/3), retrying in 2s: {e}")
            await asyncio.sleep(2)
