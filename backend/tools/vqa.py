"""
SatQuery — Mock VQA Tool

Returns realistic mock outputs for visual question answering.
Person 2 will replace the ``execute()`` internals with their
fine-tuned SatQuery-RS VLM.
"""

from __future__ import annotations

import random
from typing import Any

from backend.api.schemas import Modality, TaskType, ToolResult
from backend.tools.interfaces import BaseTool

_MOCK_ANSWERS = [
    "The image shows a residential area with several multi-story buildings.",
    "There are approximately 15 buildings visible in the scene.",
    "Yes, there is a body of water in the lower-right quadrant of the image.",
    "The primary land cover is agricultural with scattered vegetation.",
    "The image depicts an industrial zone with large warehouse structures.",
    "Three road intersections are visible in the urban area.",
    "The terrain appears to be mostly flat with gentle elevation changes.",
    "Cloud cover obscures roughly 20% of the scene.",
]


class MockVQATool(BaseTool):
    name = "vqa"
    task = TaskType.VQA
    model_name = "SatQuery-RS (mock)"
    description = "Visual question answering on remote-sensing imagery."
    required_modalities = [Modality.OPTICAL, Modality.MULTISPECTRAL, Modality.SAR]
    min_images = 1
    max_images = 1
    accepts_query = True
    output_types = ["answer"]

    async def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> ToolResult:
        question = inputs.get("question", inputs.get("query", ""))
        image_path = inputs.get("image_path", "")

        answer = random.choice(_MOCK_ANSWERS)

        return self.make_result(
            answer=answer,
            confidence=round(random.uniform(0.75, 0.96), 2),
            metadata={
                "question": question,
                "image": image_path,
                "mock": True,
            },
        )
