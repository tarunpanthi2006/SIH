"""
SatQuery — Prithvi-EO-2.0 Model Wrapper
=========================================

Wraps IBM/NASA's Prithvi-EO-2.0 (ViT-MAE) for multispectral and temporal
remote sensing analysis.

Designed to process HLS (Harmonized Landsat Sentinel-2) bands:
  - Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2
  - Resolution: 30m

Supports temporal inputs (time-series of images) via 3D patch and
positional embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PrithviEO2Model(nn.Module):
    """
    Wrapper for Prithvi-EO-2.0 foundation model.
    """

    def __init__(self, hf_model_id: str, device: str = "cpu"):
        super().__init__()
        self.hf_model_id = hf_model_id
        self.device = device
        self.model = None

    def load(self) -> None:
        """
        Loads the Prithvi-EO-2.0 model from Hugging Face.
        """
        if self.model is not None:
            return

        logger.info("Loading Prithvi-EO-2.0 (%s) to %s...", self.hf_model_id, self.device)
        try:
            # We use transformers AutoModel since Prithvi-EO-2.0 is supported in HF
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                self.hf_model_id,
                trust_remote_code=True,
            )
            self.model.to(self.device).eval()
            logger.info("Prithvi-EO-2.0 loaded successfully.")
        except ImportError:
            logger.error("transformers library is required to load Prithvi-EO-2.0")
            raise
        except Exception as e:
            logger.error("Failed to load Prithvi-EO-2.0: %s", e)
            raise

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass for feature extraction.

        Parameters
        ----------
        pixel_values : Tensor [B, C, T, H, W] or [B, C, H, W]
            Multispectral input. If no temporal dimension, a dummy one is added.

        Returns
        -------
        features : Tensor [B, N, D]
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        # Prithvi typically expects [B, C, T, H, W]
        if pixel_values.ndim == 4:
            # [B, C, H, W] -> [B, C, 1, H, W]
            pixel_values = pixel_values.unsqueeze(2)

        outputs = self.model(pixel_values, **kwargs)
        # Assuming the model returns last_hidden_state
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs

    @classmethod
    def from_pretrained(
        cls,
        hf_model_id: str = "ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
        device: str = "cpu",
    ) -> "PrithviEO2Model":
        wrapper = cls(hf_model_id, device=device)
        wrapper.load()
        return wrapper
