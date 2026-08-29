"""
SatQuery — Mock Change Detection / Change VQA Tool

Returns mock change masks, statistics, and change-based answers.
Person 3 will replace the ``execute()`` internals with ChangeFormer
and the RS-adapted VLM for change VQA.
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

_MOCK_CHANGE_ANSWERS = [
    "Significant urban expansion is visible between the two dates, with new residential structures appearing in the eastern sector.",
    "Vegetation loss is detected in the northern portion of the scene, possibly due to deforestation or agricultural conversion.",
    "A new road network has been constructed connecting the two settlement clusters.",
    "Water body extent has decreased, suggesting seasonal drought or diversion.",
    "Built-up area has increased by approximately 18% between the two acquisitions.",
]


class MockChangeDetectionTool(BaseTool):
    name = "change_detection"
    task = TaskType.CHANGE_DETECTION
    model_name = "ChangeFormer (mock)"
    description = "Bi-temporal change detection producing change masks and statistics."
    required_modalities = [Modality.OPTICAL, Modality.MULTISPECTRAL]
    min_images = 2
    max_images = 2
    accepts_query = False
    output_types = ["mask", "statistics"]

    async def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> ToolResult:
        image_a = inputs.get("image_a", "")
        image_b = inputs.get("image_b", "")
        threshold = (params or {}).get("change_threshold", 0.5)

        change_pct = round(random.uniform(5.0, 35.0), 1)
        mock_mask_path = "/artifacts/mock_change_mask.png"

        return self.make_result(
            answer=f"Change detected: {change_pct}% of the scene area has changed.",
            confidence=round(random.uniform(0.82, 0.96), 2),
            spatial_evidence=[
                SpatialEvidence(
                    type=SpatialEvidenceType.MASK,
                    path=mock_mask_path,
                    label="change_mask",
                ),
            ],
            statistics={
                "change_percentage": change_pct,
                "changed_pixels": random.randint(10000, 150000),
                "total_pixels": 512 * 512,
                "threshold": threshold,
            },
            artifacts=[mock_mask_path],
            metadata={
                "image_a": image_a,
                "image_b": image_b,
                "mock": True,
            },
        )


class MockChangeVQATool(BaseTool):
    name = "change_vqa"
    task = TaskType.CHANGE_VQA
    model_name = "ChangeFormer + SatQuery-RS (mock)"
    description = "Answer questions about changes between two temporal images."
    required_modalities = [Modality.OPTICAL, Modality.MULTISPECTRAL]
    min_images = 2
    max_images = 2
    accepts_query = True
    output_types = ["answer", "mask", "statistics"]

    async def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> ToolResult:
        question = inputs.get("question", inputs.get("query", ""))
        image_a = inputs.get("image_a", "")
        image_b = inputs.get("image_b", "")

        change_pct = round(random.uniform(5.0, 35.0), 1)
        answer = random.choice(_MOCK_CHANGE_ANSWERS)
        mock_mask_path = "/artifacts/mock_change_mask.png"

        return self.make_result(
            answer=answer,
            confidence=round(random.uniform(0.80, 0.95), 2),
            spatial_evidence=[
                SpatialEvidence(
                    type=SpatialEvidenceType.MASK,
                    path=mock_mask_path,
                    label="change_mask",
                ),
            ],
            statistics={
                "change_percentage": change_pct,
            },
            artifacts=[mock_mask_path],
            metadata={
                "question": question,
                "image_a": image_a,
                "image_b": image_b,
                "mock": True,
            },
        )
