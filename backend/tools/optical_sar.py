"""
SatQuery — Optical+SAR Cross-Modal Tool
=========================================

Agent-callable wrapper around the SkySense++ inference pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from backend.tools.contracts import SpecialistOutput, TaskType, make_error
from backend.tools.interfaces import SpecialistTool

logger = logging.getLogger(__name__)


class OpticalSarTool(SpecialistTool):
    """
    Cross-modal optical and SAR analysis tool.

    Wraps SkySense++ to jointly process optical and SAR imagery.
    Produces land-cover classification and extracts semantic features
    by fusing information from both modalities.
    """

    name = "optical_sar_analysis"
    task = TaskType.OPTICAL_SAR
    description = (
        "Perform cross-modal analysis on paired, co-registered optical and SAR images. "
        "Useful for land-cover classification and identifying objects using both modalities."
    )

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/skysensepp/",
        device: str = "cuda",
        output_dir: str = "outputs/optical_sar",
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.output_dir = output_dir

    async def load(self) -> None:
        """Pre-load the model into GPU memory."""
        from models.optical_sar.inference import _get_model
        import torch

        try:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            _get_model(self.checkpoint_path, self.device, dtype)
            self._loaded = True
            logger.info("OpticalSarTool model pre-loaded on %s", self.device)
        except Exception as exc:
            logger.error("Failed to pre-load OpticalSarTool model: %s", exc)
            self._loaded = False

    async def run(
        self,
        optical: str | None = None,
        sar: str | None = None,
        query: str | None = None,
        metadata: dict[str, Any] | None = None,
        output_dir: str | None = None,
        **kwargs: Any,
    ) -> SpecialistOutput:
        """
        Run cross-modal analysis.

        Parameters
        ----------
        optical : str
            Path to the optical image.
        sar : str
            Path to the SAR image.
        query : str, optional
            User question.
        metadata : dict, optional
            Geospatial metadata.
        output_dir : str, optional
            Override default output directory.
        """
        # ── Input validation ──
        if not optical or not sar:
            return make_error(
                TaskType.OPTICAL_SAR,
                "SkySense++",
                "Both 'optical' and 'sar' image paths are required.",
            )

        if not Path(optical).exists():
            return make_error(
                TaskType.OPTICAL_SAR,
                "SkySense++",
                f"Optical image not found: {optical}",
            )

        if not Path(sar).exists():
            return make_error(
                TaskType.OPTICAL_SAR,
                "SkySense++",
                f"SAR image not found: {sar}",
            )

        # ── Delegate to inference pipeline ──
        from models.optical_sar.inference import run_optical_sar

        result = run_optical_sar(
            optical=optical,
            sar=sar,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            output_dir=output_dir or self.output_dir,
            query=query,
            metadata=metadata,
        )

        return result
