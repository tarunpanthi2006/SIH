"""
SatQuery — Multispectral Tool
===============================

Agent-callable wrapper around the Prithvi-EO-2.0 inference pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from backend.tools.contracts import SpecialistOutput, TaskType, make_error
from backend.tools.interfaces import SpecialistTool

logger = logging.getLogger(__name__)


class MultispectralTool(SpecialistTool):
    """
    Multispectral and temporal analysis tool.
    """

    name = "multispectral_analysis"
    description = (
        "Analyze multispectral imagery (e.g., 6-band HLS data) using a foundation model. "
        "Useful for vegetation, agriculture, and general multispectral feature extraction."
    )

    def __init__(
        self,
        hf_model_id: str = "ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
        device: str = "cuda",
        output_dir: str = "outputs/multispectral",
    ):
        self.hf_model_id = hf_model_id
        self.device = device
        self.output_dir = output_dir

    async def load(self) -> None:
        """Pre-load the model into GPU memory."""
        from models.multispectral.inference import _get_model
        try:
            _get_model(self.hf_model_id, self.device)
            self._loaded = True
            logger.info("MultispectralTool model pre-loaded on %s", self.device)
        except Exception as exc:
            logger.error("Failed to pre-load MultispectralTool model: %s", exc)
            self._loaded = False

    async def run(
        self,
        image: str | None = None,
        query: str | None = None,
        metadata: dict[str, Any] | None = None,
        output_dir: str | None = None,
        **kwargs: Any,
    ) -> SpecialistOutput:
        """
        Run multispectral analysis.
        """
        if not image:
            return make_error(
                TaskType.MULTISPECTRAL,
                "Prithvi-EO-2.0",
                "Image path is required.",
            )

        if not Path(image).exists():
            return make_error(
                TaskType.MULTISPECTRAL,
                "Prithvi-EO-2.0",
                f"Image not found: {image}",
            )

        from models.multispectral.inference import run_multispectral

        result = run_multispectral(
            image=image,
            hf_model_id=self.hf_model_id,
            device=self.device,
            output_dir=output_dir or self.output_dir,
            query=query,
            metadata=metadata,
        )

        return result
