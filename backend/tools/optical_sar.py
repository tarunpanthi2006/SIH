"""
SatQuery — Mock Optical + SAR Fusion Tool

Returns mock cross-modal analysis results.
Person 3 will replace the ``execute()`` internals with SkySense++.
"""

from __future__ import annotations

import random
from typing import Any

from backend.api.schemas import Modality, TaskType, ToolResult
from backend.tools.interfaces import BaseTool

_MOCK_OPTICAL_SAR_ANSWERS = [
    "Cross-modal fusion reveals building footprints clearly in SAR backscatter, corroborated by shadows in the optical band.",
    "The optical image shows vegetation stress that correlates with low SAR coherence in the same region, suggesting waterlogging.",
    "Urban structures identified in the SAR amplitude image align with high-reflectance regions in the optical RGB composite.",
    "Ship targets detected in SAR are confirmed by corresponding signatures in the optical image under clear-sky conditions.",
    "The fused analysis indicates agricultural plots with varying moisture levels visible in SAR and crop health indicators in the optical bands.",
]


class MockOpticalSARTool(BaseTool):
    name = "optical_sar"
    task = TaskType.OPTICAL_SAR
    model_name = "SkySense++ (mock)"
    description = "Cross-modal analysis combining optical and SAR imagery."
    required_modalities = [Modality.OPTICAL, Modality.SAR]
    min_images = 2
    max_images = 2
    accepts_query = True
    output_types = ["answer"]

    async def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> ToolResult:
        optical = inputs.get("optical", inputs.get("image_a", ""))
        sar = inputs.get("sar", inputs.get("image_b", ""))
        query = inputs.get("query", inputs.get("question", ""))

        answer = random.choice(_MOCK_OPTICAL_SAR_ANSWERS)

        return self.make_result(
            answer=answer,
            confidence=round(random.uniform(0.76, 0.92), 2),
            metadata={
                "optical": optical,
                "sar": sar,
                "query": query,
                "mock": True,
            },
        )
