"""
<<<<<<< HEAD
SatQuery — Mock Grounding Tool

Returns mock bounding boxes for text-guided grounding.
Person 2 will replace the ``execute()`` internals.
=======
SatQuery Grounding Tool — Person 1 Interface
===============================================
Text-guided spatial grounding for remote sensing imagery.
Locates objects/features and returns bounding box coordinates.

Usage:
    from backend.tools.grounding import run_grounding

    result = run_grounding("path/to/satellite.png", "water body")
>>>>>>> origin/feature/vlm
"""

from __future__ import annotations

<<<<<<< HEAD
import random
from typing import Any

from backend.api.schemas import (
    Modality,
    SpatialEvidence,
    SpatialEvidenceType,
    TaskType,
    ToolResult,
)
from backend.tools.interfaces import BaseTool


class MockGroundingTool(BaseTool):
    name = "grounding"
    task = TaskType.GROUNDING
    model_name = "GeoChat (mock)"
    description = "Text-guided spatial grounding on remote-sensing imagery."
    required_modalities = [Modality.OPTICAL, Modality.MULTISPECTRAL]
    min_images = 1
    max_images = 1
    accepts_query = True
    output_types = ["answer", "bbox"]

    async def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> ToolResult:
        query = inputs.get("query", inputs.get("question", ""))
        image_path = inputs.get("image_path", "")

        # Generate 1-3 mock bounding boxes
        num_boxes = random.randint(1, 3)
        spatial: list[SpatialEvidence] = []
        for i in range(num_boxes):
            x1 = round(random.uniform(0.05, 0.5), 3)
            y1 = round(random.uniform(0.05, 0.5), 3)
            x2 = round(x1 + random.uniform(0.05, 0.3), 3)
            y2 = round(y1 + random.uniform(0.05, 0.3), 3)
            spatial.append(SpatialEvidence(
                type=SpatialEvidenceType.BBOX,
                coordinates=[x1, y1, min(x2, 1.0), min(y2, 1.0)],
                label=f"region_{i}",
                confidence=round(random.uniform(0.7, 0.95), 2),
            ))

        answer = (
            f"Found {num_boxes} region(s) matching '{query}' in the image."
        )

        return self.make_result(
            answer=answer,
            confidence=round(random.uniform(0.78, 0.94), 2),
            spatial_evidence=spatial,
            metadata={
                "query": query,
                "image": image_path,
                "num_regions": num_boxes,
                "mock": True,
            },
        )
