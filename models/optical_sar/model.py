"""
SatQuery — SkySense++ Model Wrapper
=====================================

Wraps the SkySense++ multi-modal remote-sensing foundation model for
cross-modal optical + SAR analysis.

SkySense++ uses:
  - Independent backbones for HR optical, S2 multispectral, and S1 SAR
  - A shared transformer fusion encoder for cross-modal reasoning
  - Modality-completion VAE (optional) for missing modality inference

**SAR preprocessing is NOT fake-RGB.**  SAR pixel values are radar
backscatter responses.  We apply proper dB conversion:
    dB = 10 * log10(linear_power + eps)
then normalise to the model's expected input range.

IMPORTANT — architecture status
--------------------------------
The classes below (`ModalityBackbone`, `FusionEncoder`, ...) are a
SIMPLIFIED, from-scratch stand-in for SkySense++'s actual architecture
(a much larger factorized multi-modal spatiotemporal encoder with
geo-context prototype learning — see the paper). They are NOT guaranteed
to be parameter-name- or shape-compatible with the official released
checkpoint. `from_pretrained` below verifies checkpoint/architecture
overlap and refuses to proceed on a bad match rather than silently
running inference on effectively-random weights — treat any failure
there as a signal that this class needs either real key remapping or to
be replaced with the official model code, not as something to bypass.

Both scene- and pixel-level heads consume the fused, cross-modally
attended token sequence (see `encode()`/`forward()`) — the pixel head is
NOT a modality-isolated re-run of the optical backbone.

IMPORTANT — input size is NOT a free parameter at inference time
------------------------------------------------------------------
`pos_embed` in `ModalityBackbone` is a learned parameter sized for
exactly `(img_size // patch_size) ** 2 + 1` tokens, fixed at construction
time. Feeding a tensor whose patch grid doesn't match `img_size` does NOT
raise a clear error — `forward()` slices `pos_embed[:, :x.shape[1], :]`,
which silently succeeds (wrong positions) if x has fewer patches, or
raises a plain indexing error if it has more. This means whatever size
`models/optical_sar/inference.py` resizes images to MUST equal `img_size`
here. `DEFAULT_IMG_SIZE` below is the single source of truth for that
number — inference.py imports it rather than redefining its own literal,
so the two can't drift apart independently. If you ever need a different
resolution, change it here and nowhere else.

Reference
---------
Wu et al., "A semantic-enhanced multi-modal remote sensing foundation
model for Earth observation", Nature Machine Intelligence, 2025.
GitHub: kang-wu/SkySensePlusPlus
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Single source of truth for the spatial input size this architecture was
# built for. models/optical_sar/inference.py imports this constant instead
# of hardcoding its own — see note above.
DEFAULT_IMG_SIZE = 224

# If more than this fraction of this model's parameters have no matching
# key in the checkpoint, treat the checkpoint as incompatible with this
# architecture rather than silently running on (mostly) random weights.
MAX_MISSING_KEY_FRACTION = 0.5


class SkySensePPCheckpointMismatch(RuntimeError):
    """Raised when a loaded checkpoint doesn't actually match this model's
    parameter names closely enough to trust the resulting weights."""


# ── Modality-specific backbone stubs ──────────────────────────────────────
# These mirror the SkySense++ architecture at inference level.
# Full implementation follows the official repo structure.


class ModalityBackbone(nn.Module):
    """
    A ViT-style backbone for a single modality.

    In SkySense++, each modality (HR optical, S2 multispectral, S1 SAR)
    has its own backbone.  This is a simplified version for inference.

    NOTE: `img_size` fixes the size of `pos_embed` at construction time
    (see module docstring). Inputs must be resized to exactly `img_size`
    before calling `forward` — this backbone does not resize internally.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 768,
        patch_size: int = 16,
        num_layers: int = 12,
        num_heads: int = 12,
        img_size: int = DEFAULT_IMG_SIZE,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, C, H, W]
            H and W MUST equal `self.img_size` — see class docstring.

        Returns
        -------
        features : Tensor [B, N+1, embed_dim]
            Patch tokens + CLS token.
        """
        B = x.shape[0]
        x = self.patch_embed(x)                         # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)                # [B, N, D]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                  # [B, N+1, D]
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.encoder(x)
        x = self.norm(x)
        return x


class FusionEncoder(nn.Module):
    """
    Shared transformer fusion encoder.

    Takes concatenated tokens from multiple modality backbones and
    produces a unified cross-modal representation.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 12,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        # Modality type embeddings
        self.modality_embed = nn.Embedding(3, embed_dim)  # 0=optical, 1=SAR, 2=S2

    def forward(
        self,
        tokens: list[torch.Tensor],
        modality_ids: list[int],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : list of Tensor [B, N_i, D]
        modality_ids : list of int  (0, 1, or 2)

        Returns
        -------
        fused : Tensor [B, sum(N_i), D]
        """
        tagged = []
        for tok, mid in zip(tokens, modality_ids):
            m_emb = self.modality_embed(
                torch.full((tok.shape[0], tok.shape[1]), mid,
                            dtype=torch.long, device=tok.device)
            )
            tagged.append(tok + m_emb)

        x = torch.cat(tagged, dim=1)
        x = self.encoder(x)
        return self.norm(x)


class ClassificationHead(nn.Module):
    """Simple linear classification head for land-cover analysis."""

    # Standard land-cover classes
    CLASSES = [
        "water", "built_up", "vegetation", "bare_soil",
        "agriculture", "wetland", "snow_ice", "cloud", "other",
    ]

    def __init__(self, embed_dim: int = 768, num_classes: int = 9):
        super().__init__()
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : Tensor [B, N, D]  (uses CLS token or mean pooling)

        Returns
        -------
        logits : Tensor [B, num_classes]
        """
        # Use CLS token (index 0) for classification
        cls_feat = features[:, 0, :]
        return self.head(cls_feat)


class PixelClassificationHead(nn.Module):
    """Per-pixel classification head for dense land-cover mapping."""

    CLASSES = ClassificationHead.CLASSES

    def __init__(self, embed_dim: int = 768, num_classes: int = 9,
                 patch_size: int = 16, img_size: int = DEFAULT_IMG_SIZE):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_classes = num_classes

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        patch_tokens : Tensor [B, N, D]
            Spatial patch tokens ONLY — CLS token already stripped by the
            caller. Critically, these should be tokens taken *after* the
            cross-modal fusion encoder (see SkySensePPModel.forward), so
            they carry information attended-to from the other modality —
            not a fresh, modality-isolated backbone pass.

        Returns
        -------
        class_map : Tensor [B, num_classes, H', W']
        """
        logits = self.head(patch_tokens)     # [B, N, C]
        B, N, C = logits.shape
        H = W = int(N ** 0.5)
        return logits.reshape(B, H, W, C).permute(0, 3, 1, 2)


# ── Full model ─────────────────────────────────────────────────────────────

class SkySensePPModel(nn.Module):
    """
    SkySense++ multi-modal remote-sensing foundation model.

    Supports:
    - HR optical (RGB, 3 channels)
    - S1 SAR (VV + VH, 2 channels, dB backscatter)
    - Cross-modal fusion via shared transformer encoder

    Input formats
    -------------
    optical : Tensor [B, 3, img_size, img_size]  float32  normalised [0, 1]
    sar     : Tensor [B, 2, img_size, img_size]  float32  dB normalised to [-1, 1]

    `img_size` defaults to `DEFAULT_IMG_SIZE` (see module docstring) and
    MUST match whatever spatial size the caller's preprocessing produces —
    this model does no internal resizing.

    NOT fake RGB — SAR values represent radar backscatter.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        optical_channels: int = 3,
        sar_channels: int = 2,
        img_size: int = DEFAULT_IMG_SIZE,
        patch_size: int = 16,
        num_classes: int = 9,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size

        # Modality-specific backbones
        self.optical_backbone = ModalityBackbone(
            optical_channels, embed_dim, patch_size, num_layers=12,
            num_heads=12, img_size=img_size,
        )
        self.sar_backbone = ModalityBackbone(
            sar_channels, embed_dim, patch_size, num_layers=12,
            num_heads=12, img_size=img_size,
        )

        # Cross-modal fusion
        self.fusion = FusionEncoder(embed_dim, num_layers=4, num_heads=12)

        # Task heads
        self.scene_head = ClassificationHead(embed_dim, num_classes)
        self.pixel_head = PixelClassificationHead(
            embed_dim, num_classes, patch_size, img_size,
        )

    def encode(
        self,
        optical: torch.Tensor | None = None,
        sar: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, tuple[int, int]]]:
        """
        Extract fused cross-modal features.

        At least one modality must be provided.

        Returns
        -------
        fused : Tensor [B, sum(N_i), D]
            The concatenated token sequence after cross-modal fusion.
            Token order matches concatenation order (optical tokens first
            if present, then SAR tokens) — self-attention in the fusion
            encoder does not reorder the sequence, so per-modality slices
            of the *input* order remain valid slices of the *fused*
            output, just with each token now carrying attended-to context
            from the other modality's tokens.
        token_spans : dict[str, (start, end)]
            Half-open index ranges into `fused`'s sequence dimension for
            each modality that was provided, e.g. {"optical": (0, 197)}.
            Each span includes that modality's own CLS token at `start`.
        """
        tokens = []
        modality_ids = []
        token_spans: dict[str, tuple[int, int]] = {}
        cursor = 0

        if optical is not None:
            opt_tokens = self.optical_backbone(optical)
            n = opt_tokens.shape[1]
            token_spans["optical"] = (cursor, cursor + n)
            tokens.append(opt_tokens)
            modality_ids.append(0)
            cursor += n

        if sar is not None:
            sar_tokens = self.sar_backbone(sar)
            n = sar_tokens.shape[1]
            token_spans["sar"] = (cursor, cursor + n)
            tokens.append(sar_tokens)
            modality_ids.append(1)
            cursor += n

        if not tokens:
            raise ValueError("At least one modality (optical or SAR) is required.")

        fused = self.fusion(tokens, modality_ids)
        return fused, token_spans

    def forward(
        self,
        optical: torch.Tensor | None = None,
        sar: torch.Tensor | None = None,
        task: str = "scene",
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass.

        Parameters
        ----------
        optical : Tensor [B, 3, H, W] or None
            H, W must equal `self.img_size`.
        sar : Tensor [B, 2, H, W] or None
            H, W must equal `self.img_size`.
        task : str
            "scene" for scene-level classification,
            "pixel" for dense per-pixel classification,
            "both" for both.

        Returns
        -------
        dict with keys:
            "features" — fused features [B, N, D]
            "scene_logits" — [B, num_classes]  (if task in ["scene", "both"])
            "pixel_logits" — [B, num_classes, H', W']  (if task in ["pixel", "both"])

        Notes
        -----
        Both heads now consume the POST-FUSION token sequence — pixel_logits
        is derived from the same cross-modally-attended tokens as
        scene_logits, not from a modality-isolated backbone re-run. When
        both optical and SAR are given, the spatial grid used for
        pixel_logits is anchored to the optical patch grid (SAR's own
        patch grid may differ in practice, e.g. VV/VH sensor geometry), but
        every one of those optical-grid tokens has already attended to the
        full SAR token sequence inside the fusion encoder — this is what
        makes it genuinely cross-modal rather than optical-only.

        This method performs NO resizing, reprojection, or CRS handling of
        its own — geospatial alignment is entirely the caller's
        responsibility (see models/optical_sar/inference.py::
        align_optical_sar), and spatial sizing is entirely fixed by
        `self.img_size` at construction time (see module docstring).
        """
        features, token_spans = self.encode(optical, sar)
        result = {"features": features}

        if task in ("scene", "both"):
            result["scene_logits"] = self.scene_head(features)

        if task in ("pixel", "both"):
            # Prefer the optical grid as the spatial reference (higher
            # resolution / more familiar to users as the "base" image);
            # fall back to SAR's grid if optical wasn't provided.
            spatial_modality = "optical" if "optical" in token_spans else "sar"
            start, end = token_spans[spatial_modality]
            # start is that modality's own CLS token — skip it, keep the
            # rest, which are patch tokens already fused with the other
            # modality via cross-attention in self.fusion.
            patch_tokens = features[:, start + 1:end, :]
            result["pixel_logits"] = self.pixel_head(patch_tokens)

        return result

    # ── Weight loading ─────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
        max_missing_key_fraction: float = MAX_MISSING_KEY_FRACTION,
        img_size: int = DEFAULT_IMG_SIZE,
    ) -> "SkySensePPModel":
        """
        Load SkySense++ from a checkpoint directory or file.

        Verifies how much of this model's parameters were actually
        initialized from the checkpoint and refuses to proceed if the
        overlap is too small to trust (see module docstring — this
        architecture is a simplified stand-in, not guaranteed to match
        the official released weights key-for-key).

        Parameters
        ----------
        img_size : int
            Spatial size this model instance is built for. Defaults to
            `DEFAULT_IMG_SIZE`. If you change this, whatever calls
            `preprocess_optical_array`/`preprocess_sar_array` in
            inference.py must resize to the same value, or `forward` will
            fail (see module docstring on `pos_embed`).

        Raises
        ------
        FileNotFoundError
            If no checkpoint file is found at/under the given path.
        SkySensePPCheckpointMismatch
            If too few of this model's parameters were actually
            initialized from the checkpoint.
        """
        model = cls(
            embed_dim=768, optical_channels=3, sar_channels=2,
            img_size=img_size, patch_size=16, num_classes=9,
        )
        ckpt_path = Path(checkpoint_path)

        if ckpt_path.is_dir():
            # Look for the main checkpoint file
            candidates = list(ckpt_path.glob("*.pth")) + list(ckpt_path.glob("*.pt"))
            if not candidates:
                candidates = list(ckpt_path.glob("*.safetensors"))
            if not candidates:
                raise FileNotFoundError(
                    f"No checkpoint files found in {ckpt_path}"
                )
            ckpt_file = candidates[0]
        else:
            ckpt_file = ckpt_path

        if not ckpt_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

        logger.info("Loading SkySense++ checkpoint from %s", ckpt_file)

        if ckpt_file.suffix == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(str(ckpt_file))
        else:
            state = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)

        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]

        if not isinstance(state, dict) or not state:
            raise SkySensePPCheckpointMismatch(
                f"Checkpoint at {ckpt_file} did not contain a usable "
                f"state_dict (got type {type(state)!r})."
            )

        model_keys = set(model.state_dict().keys())
        missing, unexpected = model.load_state_dict(state, strict=False)
        n_missing, n_unexpected = len(missing), len(unexpected)
        n_total = len(model_keys)
        n_loaded = n_total - n_missing

        logger.info(
            "Checkpoint match for %s: %d/%d parameters loaded "
            "(%d missing, %d unexpected keys in checkpoint)",
            ckpt_file.name, n_loaded, n_total, n_missing, n_unexpected,
        )

        missing_fraction = n_missing / n_total if n_total else 1.0
        if missing_fraction > max_missing_key_fraction:
            preview_missing = missing[:10]
            preview_unexpected = unexpected[:10]
            raise SkySensePPCheckpointMismatch(
                f"Checkpoint {ckpt_file} is incompatible with this "
                f"SkySense++ stand-in architecture: {n_missing}/{n_total} "
                f"({missing_fraction:.0%}) of this model's parameters have "
                f"no matching key in the checkpoint. Running inference "
                f"would mean using effectively-random weights while "
                f"calling the result 'SkySense++'.\n"
                f"  Sample missing keys (expected by model, absent in "
                f"checkpoint): {preview_missing}\n"
                f"  Sample unexpected keys (in checkpoint, unused by "
                f"model): {preview_unexpected}\n"
                f"This is expected if the checkpoint is the real official "
                f"SkySense++ release — its actual architecture (factorized "
                f"multi-modal spatiotemporal encoder) does not match this "
                f"simplified dual-ViT reimplementation. Either vendor the "
                f"official model code from kang-wu/SkySensePlusPlus, write "
                f"real key remapping, or train this architecture yourself "
                f"and stop presenting it as pretrained SkySense++."
            )

        if missing or unexpected:
            logger.warning(
                "Partial checkpoint match (below failure threshold): "
                "missing=%s unexpected=%s", missing[:10], unexpected[:10],
            )

        model = model.to(device=device, dtype=dtype).eval()
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(
            "SkySense++ loaded on %s (dtype=%s, %.1fM params)",
            device, dtype, param_count / 1e6,
        )
        return model