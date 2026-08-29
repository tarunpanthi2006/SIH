"""
SatQuery — Mock Caption Tool

Returns realistic mock scene descriptions.
Person 2 will replace the ``execute()`` internals.
"""

from __future__ import annotations

import random
from typing import Any

from backend.api.schemas import Modality, TaskType, ToolResult
from backend.tools.interfaces import BaseTool

_MOCK_CAPTIONS = [
    (
        "An aerial view of a suburban neighbourhood with rows of residential "
        "buildings, tree-lined streets, and a small park in the centre."
    ),
    (
        "A satellite image showing extensive agricultural fields with varying "
        "crop stages, intersected by irrigation canals and a minor road."
    ),
    (
        "A coastal scene with a sandy beach, turquoise water, a harbour with "
        "several vessels, and a built-up area extending inland."
    ),
    (
        "An industrial complex featuring large storage tanks, warehouse "
        "structures, rail lines, and adjacent vegetation patches."
    ),
    (
        "A mountainous terrain with dense forest cover, a winding river in "
        "the valley, and scattered cloud shadows across the landscape."
    ),
]


class MockCaptionTool(BaseTool):
    name = "caption"
    task = TaskType.CAPTION
    model_name = "SatQuery-RS (mock)"
    description = "Scene captioning / description for remote-sensing imagery."
    required_modalities = [Modality.OPTICAL, Modality.MULTISPECTRAL]
    min_images = 1
    max_images = 1
    accepts_query = False
    output_types = ["answer"]

    async def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> ToolResult:
        image_path = inputs.get("image_path", "")
        caption = random.choice(_MOCK_CAPTIONS)

        return self.make_result(
            answer=caption,
            confidence=round(random.uniform(0.80, 0.95), 2),
            metadata={
                "image": image_path,
                "mock": True,
            },
        )
