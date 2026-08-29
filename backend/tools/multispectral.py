"""
SatQuery — Mock Multispectral Analysis Tool

Returns mock multispectral / spectral-index analysis.
Person 3 will replace with Prithvi-EO-2.0 or similar.
"""

from __future__ import annotations

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

_MOCK_MS_ANSWERS = [
    "NDVI analysis indicates healthy vegetation (mean NDVI = 0.65) across the central region with stressed patches in the south-east.",
    "Spectral analysis reveals bare soil signatures in the western quadrant, with mixed vegetation and water in the east.",
    "Band ratio analysis detects an urban heat island effect in the densely built central area.",
    "NDWI mapping identifies a water body covering approximately 12% of the scene.",
    "Near-infrared and red-edge bands indicate crop maturity differences between the northern and southern fields.",
]


class MockMultispectralTool(BaseTool):
    name = "multispectral"
    task = TaskType.MULTISPECTRAL
    model_name = "Prithvi-EO-2.0 (mock)"
    description = "Multispectral / spectral-index analysis on remote-sensing imagery."
    required_modalities = [Modality.MULTISPECTRAL]
    min_images = 1
    max_images = 1
    accepts_query = True
    output_types = ["answer", "statistics"]

    async def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> ToolResult:
        image_path = inputs.get("image_path", "")
        query = inputs.get("query", inputs.get("question", ""))

        answer = random.choice(_MOCK_MS_ANSWERS)
        mock_index_path = "/artifacts/mock_ndvi_map.png"

        return self.make_result(
            answer=answer,
            confidence=round(random.uniform(0.78, 0.93), 2),
            spatial_evidence=[
                SpatialEvidence(
                    type=SpatialEvidenceType.MASK,
                    path=mock_index_path,
                    label="spectral_index_map",
                ),
            ],
            statistics={
                "mean_ndvi": round(random.uniform(0.3, 0.8), 2),
                "min_ndvi": round(random.uniform(-0.1, 0.1), 2),
                "max_ndvi": round(random.uniform(0.8, 0.95), 2),
            },
            artifacts=[mock_index_path],
            metadata={
                "image": image_path,
                "query": query,
                "mock": True,
            },
        )
