"""
SatQuery — Change Detection Tool
==================================

Agent-callable wrapper around the ChangeFormer inference pipeline.
Implements the ``SpecialistTool`` interface so Person 1's tool registry
can invoke it with ``await tool.run(...)``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from backend.tools.contracts import SpecialistOutput, TaskType, make_error
from backend.tools.interfaces import SpecialistTool

logger = logging.getLogger(__name__)


class ChangeTool(SpecialistTool):
    """
    Bi-temporal change detection tool.

    Wraps ChangeFormer to detect WHERE pixels changed between two
    co-registered images.  Returns a binary change mask, probability map,
    change statistics, and connected-component region analysis.

    For **change VQA** (WHAT does the change mean?), the spatial evidence
    from this tool is passed to Person 2's VLM for semantic interpretation.
    """

    name = "change_detection"
    task = TaskType.CHANGE_DETECTION
    description = (
        "Detect spatial changes between two co-registered images "
        "(bi-temporal change detection using ChangeFormer)."
    )

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/changeformer/ChangeFormer_LEVIR.pth",
        device: str = "cuda",
        threshold: float = 0.5,
        output_dir: str = "outputs/change",
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.threshold = threshold
        self.output_dir = output_dir

    async def load(self) -> None:
        """Pre-load the model into GPU memory."""
        from models.change.inference import _get_model

        try:
            _get_model(self.checkpoint_path, self.device)
            self._loaded = True
            logger.info("ChangeTool model pre-loaded on %s", self.device)
        except Exception as exc:
            logger.error("Failed to pre-load ChangeTool model: %s", exc)
            self._loaded = False

    async def run(
        self,
        image_a: str | None = None,
        image_b: str | None = None,
        query: str | None = None,
        metadata: dict[str, Any] | None = None,
        threshold: float | None = None,
        output_dir: str | None = None,
        **kwargs: Any,
    ) -> SpecialistOutput:
        """
        Run change detection on a bi-temporal image pair.

        Parameters
        ----------
        image_a : str
            Path to the earlier image.
        image_b : str
            Path to the later image.
        query : str, optional
            User question (stored in metadata for downstream VQA).
        metadata : dict, optional
            Geospatial metadata (e.g. ``{"gsd": 0.5}`` for 0.5 m/pixel).
        threshold : float, optional
            Override default probability threshold.
        output_dir : str, optional
            Override default output directory.

        Returns
        -------
        SpecialistOutput
            Structured output with change mask, statistics, etc.
        """
        # ── Input validation ──
        if not image_a or not image_b:
            return make_error(
                TaskType.CHANGE_DETECTION,
                "ChangeFormer",
                "Both image_a and image_b are required for change detection.",
            )

        if not Path(image_a).exists():
            return make_error(
                TaskType.CHANGE_DETECTION,
                "ChangeFormer",
                f"image_a not found: {image_a}",
            )

        if not Path(image_b).exists():
            return make_error(
                TaskType.CHANGE_DETECTION,
                "ChangeFormer",
                f"image_b not found: {image_b}",
            )

        # ── Attach query to metadata for downstream VQA ──
        meta = dict(metadata or {})
        if query:
            meta["user_query"] = query

        # ── Delegate to inference pipeline ──
        from models.change.inference import run_change

        result = run_change(
            image_a=image_a,
            image_b=image_b,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            threshold=threshold or self.threshold,
            output_dir=output_dir or self.output_dir,
            metadata=meta,
        )

        return result
