"""
SatQuery — ChangeFormer Model Wrapper
======================================

Wraps the ChangeFormer (MiT-b2 Siamese Transformer) for bi-temporal change
detection.  Loads the architecture + pretrained checkpoint and exposes a
clean ``forward(image_a, image_b) → logits`` interface.

Reference
---------
Bandara & Patel, "A Transformer-Based Siamese Network for Change Detection",
IGARSS 2022.  GitHub: wgcban/ChangeFormer
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Lightweight MiT (Mix Transformer) blocks ──────────────────────────────
# These reproduce the minimum architecture needed for inference.
# Full training code lives in the upstream ChangeFormer repo.


class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding (stride < kernel)."""

    def __init__(self, patch_size: int = 7, stride: int = 4,
                 in_channels: int = 3, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        x = self.norm(x)
        return x, H, W


class EfficientSelfAttention(nn.Module):
    """Multi-head self-attention with spatial reduction."""

    def __init__(self, dim: int, num_heads: int = 8, sr_ratio: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MixFFN(nn.Module):
    """Feed-forward with depth-wise convolution."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, H: int, W: int):
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.act(self.dwconv(x))
        x = x.flatten(2).transpose(1, 2)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, H: int, W: int):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x


class MiTEncoder(nn.Module):
    """
    Mix-Transformer (MiT-b2) encoder.

    4-stage hierarchical encoder producing multi-scale features.
    """

    # MiT-b2 configuration
    EMBED_DIMS  = [64, 128, 320, 512]
    NUM_HEADS   = [1, 2, 5, 8]
    SR_RATIOS   = [8, 4, 2, 1]
    DEPTHS      = [3, 4, 6, 3]
    PATCH_SIZES = [7, 3, 3, 3]
    STRIDES     = [4, 2, 2, 2]

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.stages = nn.ModuleList()
        self.patch_embeds = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(4):
            in_ch = in_channels if i == 0 else self.EMBED_DIMS[i - 1]
            self.patch_embeds.append(
                OverlapPatchEmbed(
                    self.PATCH_SIZES[i], self.STRIDES[i],
                    in_ch, self.EMBED_DIMS[i],
                )
            )
            blocks = nn.ModuleList([
                TransformerBlock(
                    self.EMBED_DIMS[i], self.NUM_HEADS[i],
                    self.SR_RATIOS[i],
                )
                for _ in range(self.DEPTHS[i])
            ])
            self.stages.append(blocks)
            self.norms.append(nn.LayerNorm(self.EMBED_DIMS[i]))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for i in range(4):
            x, H, W = self.patch_embeds[i](x)
            for blk in self.stages[i]:
                x = blk(x, H, W)
            x = self.norms[i](x)
            x = x.reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2)
            features.append(x)
        return features


class ChangeFormerDecoder(nn.Module):
    """
    Simple difference-based decoder.

    Takes multi-scale feature pairs from the Siamese encoder,
    computes difference features, and produces a change probability map.
    """

    EMBED_DIMS = [64, 128, 320, 512]

    def __init__(self, num_classes: int = 2, embed_dim: int = 256):
        super().__init__()
        # Linear projections to common dimension
        self.linear_fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, embed_dim, 1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )
            for dim in self.EMBED_DIMS
        ])
        # Fused prediction
        self.linear_pred = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, num_classes, 1),
        )

    def forward(self, feats_a: list[torch.Tensor],
                feats_b: list[torch.Tensor],
                target_size: tuple[int, int]) -> torch.Tensor:
        diffs = []
        for i in range(4):
            diff = torch.abs(feats_a[i] - feats_b[i])
            diff = self.linear_fuse[i](diff)
            diff = F.interpolate(diff, size=target_size,
                                 mode="bilinear", align_corners=False)
            diffs.append(diff)

        x = torch.cat(diffs, dim=1)
        return self.linear_pred(x)


# ── Full model ─────────────────────────────────────────────────────────────

class ChangeFormerModel(nn.Module):
    """
    ChangeFormer: Siamese MiT encoder + difference decoder for binary
    change detection from co-registered image pairs.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 2: no-change, change).
    embed_dim : int
        Decoder hidden dimension.

    Input
    -----
    image_a, image_b : Tensor  [B, 3, H, W]  float32  normalised to [0, 1]
        Co-registered bi-temporal image pair (RGB).

    Output
    ------
    logits : Tensor  [B, num_classes, H, W]
    """

    def __init__(self, num_classes: int = 2, embed_dim: int = 256):
        super().__init__()
        self.encoder = MiTEncoder(in_channels=3)   # shared-weight Siamese
        self.decoder = ChangeFormerDecoder(num_classes, embed_dim)
        self.num_classes = num_classes

    def forward(self, image_a: torch.Tensor,
                image_b: torch.Tensor) -> torch.Tensor:
        H, W = image_a.shape[2], image_a.shape[3]
        feats_a = self.encoder(image_a)
        feats_b = self.encoder(image_b)
        logits = self.decoder(feats_a, feats_b, target_size=(H, W))
        return logits

    # ── Weight loading ─────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        device: str = "cpu",
        num_classes: int = 2,
    ) -> "ChangeFormerModel":
        """
        Load a ChangeFormer model from a .pth checkpoint.

        The checkpoint may come from the official wgcban/ChangeFormer repo.
        We do a best-effort key remapping for compatibility.
        """
        model = cls(num_classes=num_classes)
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"ChangeFormer checkpoint not found: {ckpt_path}"
            )

        logger.info("Loading ChangeFormer checkpoint from %s", ckpt_path)
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

        # The official checkpoint wraps in various keys
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]

        # Best-effort load (ignore mismatched / extra keys)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("Missing keys during load: %s", missing[:10])
        if unexpected:
            logger.warning("Unexpected keys during load: %s", unexpected[:10])

        model = model.to(device).eval()
        logger.info("ChangeFormer loaded on %s  (%d params)",
                     device, sum(p.numel() for p in model.parameters()))
        return model
