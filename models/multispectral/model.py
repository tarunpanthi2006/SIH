"""
SatQuery — Prithvi-EO-2.0 Model Wrapper
=========================================

Wraps IBM/NASA's Prithvi-EO-2.0 geospatial foundation model.

Official loading path is via TerraTorch:
    from terratorch.registry import BACKBONE_REGISTRY
    model = BACKBONE_REGISTRY.build("prithvi_eo_v2_600", pretrained=True)

Architecture
------------
ViT-MAE with 3D patch + positional embeddings.
Input tensor shape: [B, T, C, H, W]
  B = batch size
  T = number of time steps (use T=1 for single-image inference)
  C = 6 spectral channels (Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2)
  H, W = spatial dimensions (must be divisible by patch size)

The 600M variant (prithvi_eo_v2_600) does NOT have temporal/location
embeddings — those are in the *-TL variants. For temporal analysis use
"prithvi_eo_v2_600_tl" instead and provide timestamps.

Checkpoint
----------
HuggingFace: ibm-nasa-geospatial/Prithvi-EO-2.0-600M
Local (after download): checkpoints/prithvi/

VRAM estimate
-------------
~8-12 GB on float32, ~6-8 GB on float16.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Backbone registry name → HuggingFace model ID
BACKBONE_CONFIGS = {
    "prithvi_eo_v2_600": "ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
    "prithvi_eo_v2_600_tl": "ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL",
    "prithvi_eo_v2_300": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
    "prithvi_eo_v2_300_tl": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL",
}


class PrithviEO2Model(nn.Module):
    """
    Thin wrapper around Prithvi-EO-2.0 backbone for feature extraction.

    Loads via TerraTorch (primary path) with a fallback to HuggingFace
    `AutoModel` if TerraTorch is not installed.

    Parameters
    ----------
    backbone_name : str
        TerraTorch backbone registry name, e.g. "prithvi_eo_v2_600".
    local_dir : str or Path, optional
        If provided, loads weights from this local directory (after running
        download_models.py) instead of downloading from HuggingFace.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        backbone_name: str = "prithvi_eo_v2_600",
        local_dir: str | Path | None = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.local_dir = Path(local_dir) if local_dir else None
        self.device = device
        self.backbone: nn.Module | None = None

    @staticmethod
    def _find_weight_file(directory: Path) -> Path | None:
        """Find the actual weight file inside a checkpoint directory.

        ``download_models.py`` uses ``huggingface_hub.snapshot_download``,
        which produces a directory containing config files, READMEs, **and**
        the real weight file(s).  TerraTorch's ``ckpt_path`` parameter
        expects a *file*, not a directory — so we need to locate the
        specific weight file.
        """
        for pattern in ("*.safetensors", "*.pt", "*.pth", "*.bin"):
            candidates = list(directory.rglob(pattern))
            if candidates:
                # Prefer the largest file (weight files dwarf configs)
                return max(candidates, key=lambda p: p.stat().st_size)
        return None

    def load(self) -> None:
        """Load the backbone. Call before any forward pass."""
        if self.backbone is not None:
            return

        logger.info(
            "Loading Prithvi-EO-2.0 backbone: %s on %s",
            self.backbone_name, self.device,
        )

        # ── Primary path: TerraTorch ───────────────────────────────────
        try:
            from terratorch.registry import BACKBONE_REGISTRY  # type: ignore

            build_kwargs: dict = {"pretrained": True}

            # If a local directory exists, find the actual weight file
            # inside it.  ckpt_path must be a FILE, not a directory.
            if self.local_dir and self.local_dir.exists():
                weight_file = self._find_weight_file(self.local_dir)
                if weight_file:
                    logger.info(
                        "Loading from local weight file: %s", weight_file
                    )
                    build_kwargs["ckpt_path"] = str(weight_file)
                else:
                    logger.info(
                        "No weight file found in %s; "
                        "letting TerraTorch use its HuggingFace cache.",
                        self.local_dir,
                    )

            self.backbone = BACKBONE_REGISTRY.build(
                self.backbone_name, **build_kwargs
            )

            self.backbone = self.backbone.to(self.device).eval()
            n_params = sum(p.numel() for p in self.backbone.parameters())
            logger.info(
                "Prithvi-EO-2.0 loaded via TerraTorch (%.0fM params)",
                n_params / 1e6,
            )
            return

        except ImportError:
            logger.warning(
                "TerraTorch not installed. "
                "Falling back to HuggingFace transformers AutoModel. "
                "Install TerraTorch for the official loading path: "
                "pip install terratorch"
            )
        except Exception as exc:
            logger.warning(
                "TerraTorch loading failed (%s). Trying HF fallback.", exc
            )

        # ── Fallback path: HuggingFace AutoModel ──────────────────────
        # NOTE: Prithvi-EO-2.0 may not be compatible with AutoModel
        # directly if the repo doesn't ship a `config.json` that maps
        # to a standard HF model class. If this also fails, the error
        # message will tell you to install terratorch.
        try:
            from transformers import AutoModel  # type: ignore

            hf_id = BACKBONE_CONFIGS.get(
                self.backbone_name,
                "ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
            )
            load_path = str(self.local_dir) if self.local_dir else hf_id

            logger.info(
                "Loading Prithvi-EO-2.0 via HF AutoModel from: %s", load_path
            )
            self.backbone = AutoModel.from_pretrained(
                load_path,
                trust_remote_code=True,
            )
            self.backbone = self.backbone.to(self.device).eval()
            logger.info("Prithvi-EO-2.0 loaded via HF AutoModel.")

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Prithvi-EO-2.0 via both TerraTorch and "
                f"HuggingFace AutoModel.\n"
                f"  TerraTorch error: see log above.\n"
                f"  HuggingFace error: {exc}\n"
                f"Install TerraTorch to use the official loading path:\n"
                f"  pip install terratorch\n"
                f"Then re-run: python scripts/download_models.py"
            ) from exc

    def forward(
        self,
        pixel_values: torch.Tensor,
        timestamps: torch.Tensor | None = None,
        location: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Extract features from multispectral input.

        Parameters
        ----------
        pixel_values : Tensor [B, C, T, H, W]
            Multispectral input.
            B = batch, C = 6 channels, T = time steps, H × W = spatial.
            For single-image inference, use T=1.
        timestamps : Tensor [B, T, 2], optional
            Year + day-of-year for each time step. Only meaningful for
            *-TL backbone variants; ignored by the 600M (non-TL) model.
        location : Tensor [B, 2], optional
            Centre latitude and longitude. Only meaningful for -TL variants.

        Returns
        -------
        features : Tensor [B, N, D]
            Patch token embeddings from the backbone encoder.
        """
        if self.backbone is None:
            raise RuntimeError(
                "Model not loaded. Call load() before forward()."
            )

        # TerraTorch backbones typically accept (pixel_values,) with
        # optional kwargs for temporal/location encodings.
        try:
            kwargs: dict = {}
            if timestamps is not None:
                kwargs["timestamps"] = timestamps
            if location is not None:
                kwargs["location"] = location

            out = self.backbone(pixel_values, **kwargs)

            # TerraTorch backbones return either a tensor or a list of
            # tensors (one per stage). We want the final encoder output.
            if isinstance(out, (list, tuple)):
                return out[-1]   # last stage features
            return out

        except Exception as exc:
            raise RuntimeError(
                f"Prithvi-EO-2.0 forward pass failed: {exc}\n"
                f"Input shape: {pixel_values.shape}. "
                f"Expected [B, T, 6, H, W] where H and W are "
                f"multiples of the model's patch size (typically 16)."
            ) from exc

    @classmethod
    def from_pretrained(
        cls,
        backbone_name: str = "prithvi_eo_v2_600",
        local_dir: str | Path | None = "checkpoints/prithvi",
        device: str = "cpu",
    ) -> "PrithviEO2Model":
        """
        Convenience constructor that immediately calls load().

        Parameters
        ----------
        backbone_name : str
            TerraTorch registry name. Options:
              "prithvi_eo_v2_600"     — 600M params, no temporal/location
              "prithvi_eo_v2_600_tl"  — 600M params + temporal + location
              "prithvi_eo_v2_300"     — 300M params
              "prithvi_eo_v2_300_tl"  — 300M params + temporal + location
        local_dir : path, optional
            Local checkpoint directory (from download_models.py).
            Defaults to "checkpoints/prithvi".
        device : str
            "cuda" or "cpu".
        """
        instance = cls(
            backbone_name=backbone_name,
            local_dir=local_dir,
            device=device,
        )
        instance.load()
        return instance
