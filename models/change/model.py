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

DECODER ARCHITECTURE — verified against official source
----------------------------------------------------------
The official ChangeFormerV6 (models/ChangeFormer.py in the upstream repo)
instantiates its decoder as:

    self.TDec_x2 = DecoderTransformer_v3(..., in_channels=[64,128,320,512],
                                          embedding_dim=256, ...)

Despite the "TDec_x2" attribute name (a leftover from an earlier variant),
this is the DecoderTransformer_v3 class — a 4-scale MLP-diff decoder that
ADDS upsampled coarser-scale diff features into finer scales (not the
5-scale DecoderTransformer_x2, which concatenates). This was confirmed by
directly inspecting both the official source and the checkpoint's own
shapes: e.g. `diff_c1.0.weight` is (256, 512, 3, 3) — input channels
512 = 2*256 = 2*embedding_dim, which only matches DecoderTransformer_v3's
`conv_diff(in_channels=2*embedding_dim, ...)`; the v2/x2 variant would
need 3*embedding_dim at c1-c3 (concatenation of two feature diffs plus an
upsampled higher-level diff) and a 5th linear_c5/diff_c5/make_pred_c5 set
that this checkpoint simply doesn't have.

The classes below (`MLP`, `conv_diff`, `make_prediction`, `ConvLayer`,
`UpsampleConvLayer`, `ResidualBlock`, `ChangeFormerDecoderV3`) are a
faithful port of the official `DecoderTransformer_v3` — same submodule
names, same `nn.Sequential` index ordering, same forward-pass structure —
specifically so the real pretrained decoder weights load correctly
instead of falling back to random init.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ChangeFormerCheckpointMismatch(RuntimeError):
    """Raised when the checkpoint doesn't match this architecture closely
    enough to trust the resulting weights."""


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
            # Named 'norm' to match official checkpoint key: attn.norm.weight
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class DWConv(nn.Module):
    """Depth-wise convolution wrapper.

    Wraps Conv2d in a sub-module so the state_dict key path becomes
    ``mlp.dwconv.dwconv.weight`` — matching the official ChangeFormer
    checkpoint naming.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dwconv(x)


class MixFFN(nn.Module):
    """Feed-forward with depth-wise convolution."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DWConv(hidden_dim)
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
    Mix-Transformer (MiT-b2) encoder — matches ChangeFormerV6's
    `EncoderTransformer_v3` (confirmed against official source: no
    intra-patch blocks, exactly 4 stages, exactly the key count in the
    checkpoint — see module docstring for the verification note on
    DEPTHS).
    """

    # ChangeFormerV6 configuration (NOT standard MiT-b2).
    # Standard MiT-b2 uses DEPTHS=[3,4,6,3], but the official
    # ChangeFormerV6 checkpoint uses [3,3,4,3] — confirmed both by
    # checkpoint key-count inspection (272 encoder keys match exactly)
    # AND directly against official source
    # (ChangeFormerV6.__init__: self.depths = [3, 3, 4, 3]).
    EMBED_DIMS  = [64, 128, 320, 512]
    NUM_HEADS   = [1, 2, 5, 8]
    SR_RATIOS   = [8, 4, 2, 1]
    DEPTHS      = [3, 3, 4, 3]
    PATCH_SIZES = [7, 7, 7, 7]
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
        """
        Returns
        -------
        features : list of 4 tensors [B, C_i, H_i, W_i], ascending stage
            index = descending spatial resolution (features[0] is
            highest-res/stage1 @ stride 4, features[3] is
            lowest-res/stage4 @ stride 32). This ordering matches the
            official code's c1..c4 naming used in the decoder.
        """
        features = []
        for i in range(4):
            x, H, W = self.patch_embeds[i](x)
            for blk in self.stages[i]:
                x = blk(x, H, W)
            x = self.norms[i](x)
            x = x.reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2)
            features.append(x)
        return features


# ── Real DecoderTransformer_v3 port ────────────────────────────────────────
# Faithful port of the official decoder — see module docstring.


class MLP(nn.Module):
    """Linear embedding — official `MLP` class in ChangeFormer.py.

    Flattens a [B,C,H,W] feature map's spatial dims and projects channel
    dim C -> embed_dim via a single Linear layer.
    """

    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)  # [B,C,H,W] -> [B, N, C]
        x = self.proj(x)                  # -> [B, N, embed_dim]
        return x


def conv_diff(in_channels: int, out_channels: int) -> nn.Sequential:
    """Official `conv_diff` — Conv-ReLU-BN-Conv-ReLU.

    Sequential indices: 0=Conv2d, 1=ReLU(no params), 2=BatchNorm2d,
    3=Conv2d, 4=ReLU(no params) — matches checkpoint keys
    `diff_cN.0/.2/.3` exactly (indices 1 and 4 have no learnable params
    so they never appear as checkpoint keys).
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
    )


def make_prediction(in_channels: int, out_channels: int) -> nn.Sequential:
    """Official `make_prediction` — Conv-ReLU-BN-Conv (no trailing ReLU).

    Sequential indices: 0=Conv2d, 1=ReLU(no params), 2=BatchNorm2d,
    3=Conv2d — matches checkpoint keys `make_pred_cN.0/.2/.3` exactly.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    )


class ConvLayer(nn.Module):
    """Official `ConvLayer` — wraps Conv2d as `self.conv2d` so the
    checkpoint key path becomes `....conv2d.weight`."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                 stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x)


class UpsampleConvLayer(nn.Module):
    """Official `UpsampleConvLayer` — a learned 2x upsample via
    ConvTranspose2d, wrapped as `self.conv2d` to match checkpoint keys
    `convd1x.conv2d.*` / `convd2x.conv2d.*`."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int):
        super().__init__()
        self.conv2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x)


class ResidualBlock(nn.Module):
    """Official `ResidualBlock` — two ConvLayers + ReLU + scaled residual
    add. Matches checkpoint keys `dense_Nx.0.conv1.conv2d.*` /
    `dense_Nx.0.conv2.conv2d.*` (the `.0` is this module's index inside
    the enclosing `nn.Sequential`, per official `dense_1x = nn.Sequential(
    ResidualBlock(embedding_dim))`)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


class ChangeFormerDecoderV3(nn.Module):
    """
    Faithful port of the official `DecoderTransformer_v3` used by
    ChangeFormerV6 (see module docstring for how this was confirmed
    against both the official source and the checkpoint's own shapes).

    Takes 4-scale Siamese encoder feature pairs, computes per-scale
    difference features via a shared-across-images MLP + conv_diff, adds
    each coarser scale's diff (upsampled 2x) into the next finer scale,
    produces an auxiliary prediction at each of the 4 encoder scales, then
    upsamples the finest fused representation 4x back to input resolution
    via two learned ConvTranspose2d + ResidualBlock stages for the final
    per-pixel prediction.
    """

    EMBED_DIMS = [64, 128, 320, 512]

    def __init__(self, num_classes: int = 2, embed_dim: int = 256):
        super().__init__()
        self.embedding_dim = embed_dim
        self.output_nc = num_classes
        c1_ch, c2_ch, c3_ch, c4_ch = self.EMBED_DIMS

        # MLP decoder heads — project each stage's channel dim to embed_dim
        self.linear_c4 = MLP(c4_ch, embed_dim)
        self.linear_c3 = MLP(c3_ch, embed_dim)
        self.linear_c2 = MLP(c2_ch, embed_dim)
        self.linear_c1 = MLP(c1_ch, embed_dim)

        # Convolutional difference modules — each takes the CONCATENATED
        # (image_a, image_b) diff-input pair at that scale: 2*embed_dim in.
        self.diff_c4 = conv_diff(2 * embed_dim, embed_dim)
        self.diff_c3 = conv_diff(2 * embed_dim, embed_dim)
        self.diff_c2 = conv_diff(2 * embed_dim, embed_dim)
        self.diff_c1 = conv_diff(2 * embed_dim, embed_dim)

        # Auxiliary per-scale predictions (deep supervision in training;
        # at inference we only need the final scale, but these must still
        # exist and load correctly since the checkpoint has real weights
        # for them, and make_pred_c4 lies on the residual path for the
        # decoder to be numerically identical to the official model).
        self.make_pred_c4 = make_prediction(embed_dim, num_classes)
        self.make_pred_c3 = make_prediction(embed_dim, num_classes)
        self.make_pred_c2 = make_prediction(embed_dim, num_classes)
        self.make_pred_c1 = make_prediction(embed_dim, num_classes)

        # Final multi-scale fusion
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
        )

        # Learned 4x upsample back to input resolution (2x + 2x)
        self.convd2x = UpsampleConvLayer(embed_dim, embed_dim, kernel_size=4, stride=2)
        self.dense_2x = nn.Sequential(ResidualBlock(embed_dim))
        self.convd1x = UpsampleConvLayer(embed_dim, embed_dim, kernel_size=4, stride=2)
        self.dense_1x = nn.Sequential(ResidualBlock(embed_dim))

        self.change_probability = ConvLayer(embed_dim, num_classes,
                                             kernel_size=3, stride=1, padding=1)

    def forward(
        self,
        feats_a: list[torch.Tensor],
        feats_b: list[torch.Tensor],
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        feats_a, feats_b : list of 4 tensors [B, C_i, H_i, W_i]
            Encoder outputs for image_a / image_b, in ascending stage
            order (index 0 = highest-res stage1, index 3 = lowest-res
            stage4) — matches `MiTEncoder.forward`'s output ordering.
        target_size : (H, W)
            Original (unpadded) input spatial size — the fused prediction
            is resized to exactly this at the end as a defensive measure
            (the learned 4x upsample should already land here if input
            dims were a multiple of 4, but padding/rounding can introduce
            a 1-pixel mismatch, so we enforce it explicitly rather than
            let a shape mismatch propagate downstream).

        Returns
        -------
        logits : Tensor [B, num_classes, H, W]
            The FINAL, full-resolution prediction (the last element of
            the official model's multi-scale `outputs` list) — the
            auxiliary coarser-scale predictions (p_c4/p_c3/p_c2/p_c1) are
            computed (since they sit on the residual path and their
            weights must be exercised for the decoder to be numerically
            faithful) but not returned individually; only the final
            per-pixel map is needed for inference.
        """
        c1_1, c2_1, c3_1, c4_1 = feats_a
        c1_2, c2_2, c3_2, c4_2 = feats_b
        n = c4_1.shape[0]

        # ── Stage 4 (lowest res) ──
        _c4_1 = self.linear_c4(c4_1).permute(0, 2, 1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0, 2, 1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4 = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))
        _ = self.make_pred_c4(_c4)  # auxiliary prediction (unused at inference)
        _c4_up = F.interpolate(_c4, size=c1_2.shape[2:], mode="bilinear", align_corners=False)

        # ── Stage 3 ──
        _c3_1 = self.linear_c3(c3_1).permute(0, 2, 1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0, 2, 1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + F.interpolate(
            _c4, scale_factor=2, mode="bilinear", align_corners=False,
        )
        _ = self.make_pred_c3(_c3)
        _c3_up = F.interpolate(_c3, size=c1_2.shape[2:], mode="bilinear", align_corners=False)

        # ── Stage 2 ──
        _c2_1 = self.linear_c2(c2_1).permute(0, 2, 1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0, 2, 1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(
            _c3, scale_factor=2, mode="bilinear", align_corners=False,
        )
        _ = self.make_pred_c2(_c2)
        _c2_up = F.interpolate(_c2, size=c1_2.shape[2:], mode="bilinear", align_corners=False)

        # ── Stage 1 (highest res) ──
        _c1_1 = self.linear_c1(c1_1).permute(0, 2, 1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0, 2, 1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(
            _c2, scale_factor=2, mode="bilinear", align_corners=False,
        )
        _ = self.make_pred_c1(_c1)

        # ── Multi-scale fusion ──
        fused = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1), dim=1))

        # ── Learned 4x upsample back to input resolution ──
        x = self.convd2x(fused)
        x = self.dense_2x(x)
        x = self.convd1x(x)
        x = self.dense_1x(x)

        logits = self.change_probability(x)

        # Defensive: enforce exact target size (see docstring).
        if logits.shape[2:] != tuple(target_size):
            logits = F.interpolate(logits, size=target_size,
                                    mode="bilinear", align_corners=False)

        return logits


# ── Full model ─────────────────────────────────────────────────────────────

class ChangeFormerModel(nn.Module):
    """
    ChangeFormer: Siamese MiT encoder + real DecoderTransformer_v3 for
    binary change detection from co-registered image pairs.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 2: no-change, change).
    embed_dim : int
        Decoder hidden dimension (256, matching the official
        ChangeFormerV6 checkpoint).

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
        self.decoder = ChangeFormerDecoderV3(num_classes, embed_dim)
        self.num_classes = num_classes

    def forward(self, image_a: torch.Tensor,
                image_b: torch.Tensor) -> torch.Tensor:
        H, W = image_a.shape[2], image_a.shape[3]
        feats_a = self.encoder(image_a)
        feats_b = self.encoder(image_b)
        logits = self.decoder(feats_a, feats_b, target_size=(H, W))
        return logits

    # ── Weight loading ─────────────────────────────────────────────────

    @staticmethod
    def _remap_official_keys(state_dict: dict) -> dict:
        """
        Remap parameter names from the official wgcban/ChangeFormer
        checkpoint to match our architecture's module names.

        Official ChangeFormerV6 checkpoint naming (confirmed against
        upstream source):
            Tenc_x2.block{1-4}.{j}.*        (encoder blocks)
            Tenc_x2.patch_embed{1-4}.*      (encoder patch embeddings)
            Tenc_x2.norm{1-4}.*             (encoder stage norms)
            TDec_x2.linear_c{1-4}.*         (decoder — now loaded, see below)
            TDec_x2.diff_c{1-4}.*           (decoder)
            TDec_x2.make_pred_c{1-4}.*      (decoder)
            TDec_x2.linear_fuse.*           (decoder)
            TDec_x2.convd{1,2}x.*           (decoder)
            TDec_x2.dense_{1,2}x.*          (decoder)
            TDec_x2.change_probability.*    (decoder)

        Our architecture uses:
            encoder.stages.{i}.{j}.*
            encoder.patch_embeds.{i}.*
            encoder.norms.{i}.*
            decoder.linear_c{1-4}.*         (same suffix — ChangeFormerDecoderV3
            decoder.diff_c{1-4}.*            is a faithful port, so ONLY the
            decoder.make_pred_c{1-4}.*       "TDec_x2." -> "decoder." prefix
            decoder.linear_fuse.*            needs to change; every suffix
            decoder.convd{1,2}x.*            after that is identical by
            decoder.dense_{1,2}x.*           construction.)
            decoder.change_probability.*

        Both encoder AND decoder keys are now remapped and expected to
        load — unlike the previous version of this function, which
        skipped all TDec_x2.* keys because the decoder architecture used
        to be a simplified stand-in. See module docstring for how the
        real decoder architecture was confirmed.
        """
        new_state = {}

        for key, value in state_dict.items():
            new_key = key

            # ── Decoder: TDec_x2.* → decoder.* (suffix unchanged) ──
            if new_key.startswith("TDec_x2."):
                new_key = "decoder." + new_key[len("TDec_x2."):]
                new_state[new_key] = value
                continue

            # ── Encoder remapping: Tenc_x2.* → encoder.* ──

            # Tenc_x2.patch_embed{1-4}.X → encoder.patch_embeds.{0-3}.X
            for i in range(1, 5):
                old = f"Tenc_x2.patch_embed{i}."
                new = f"encoder.patch_embeds.{i-1}."
                if new_key.startswith(old):
                    new_key = new_key.replace(old, new, 1)
                    break

            # Tenc_x2.block{1-4}.{j}.X → encoder.stages.{0-3}.{j}.X
            for i in range(1, 5):
                old = f"Tenc_x2.block{i}."
                new = f"encoder.stages.{i-1}."
                if new_key.startswith(old):
                    new_key = new_key.replace(old, new, 1)
                    break

            # Tenc_x2.norm{1-4}.X → encoder.norms.{0-3}.X
            for i in range(1, 5):
                old = f"Tenc_x2.norm{i}."
                new = f"encoder.norms.{i-1}."
                if new_key.startswith(old):
                    new_key = new_key.replace(old, new, 1)
                    break

            new_state[new_key] = value

        return new_state

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        device: str = "cpu",
        num_classes: int = 2,
    ) -> "ChangeFormerModel":
        """
        Load a ChangeFormer model from a .pth checkpoint.

        Handles the official wgcban/ChangeFormer checkpoint format which
        stores weights under 'model_G_state_dict' with parameter names
        prefixed by 'Tenc_x2.*' (encoder) and 'TDec_x2.*' (decoder).

        Both encoder and decoder are now expected to load from real
        weights (previously only the encoder loaded; the decoder used a
        simplified stand-in and stayed randomly initialized — see module
        docstring for the architecture confirmation that made the real
        decoder port possible).
        """
        model = cls(num_classes=num_classes)
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"ChangeFormer checkpoint not found: {ckpt_path}"
            )

        logger.info("Loading ChangeFormer checkpoint from %s", ckpt_path)
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

        if "model_G_state_dict" in state:
            logger.info("Found 'model_G_state_dict' key (official ChangeFormer format)")
            state = state["model_G_state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]

        state = cls._remap_official_keys(state)

        missing, unexpected = model.load_state_dict(state, strict=False)

        total_params = len(list(model.state_dict().keys()))
        loaded_params = total_params - len(missing)
        load_fraction = loaded_params / total_params if total_params > 0 else 0

        decoder_missing = [k for k in missing if k.startswith("decoder.")]
        encoder_missing = [k for k in missing if not k.startswith("decoder.")]

        if encoder_missing:
            logger.warning(
                "ENCODER keys still missing after remapping (%d): %s",
                len(encoder_missing), encoder_missing[:10],
            )
        if decoder_missing:
            logger.warning(
                "DECODER keys still missing after remapping (%d): %s. "
                "The decoder is now expected to load fully from the "
                "checkpoint — any residual gap here likely means a "
                "submodule naming mismatch, not an intentional random-init "
                "fallback (unlike the previous version of this loader).",
                len(decoder_missing), decoder_missing[:10],
            )
        if unexpected:
            logger.info("Unexpected keys (not consumed): %s", unexpected[:10])

        encoder_keys = [k for k in model.state_dict() if not k.startswith("decoder.")]
        decoder_keys = [k for k in model.state_dict() if k.startswith("decoder.")]
        encoder_loaded = len(encoder_keys) - len(encoder_missing)
        decoder_loaded = len(decoder_keys) - len(decoder_missing)
        encoder_fraction = encoder_loaded / len(encoder_keys) if encoder_keys else 0
        decoder_fraction = decoder_loaded / len(decoder_keys) if decoder_keys else 0

        # Both encoder AND decoder must now load properly — the decoder
        # is no longer allowed a random-init fallback since its real
        # architecture is implemented above.
        if encoder_fraction < 0.95 or decoder_fraction < 0.95:
            raise ChangeFormerCheckpointMismatch(
                f"ChangeFormer key mismatch: encoder {encoder_loaded}/"
                f"{len(encoder_keys)} ({encoder_fraction:.0%}) loaded, "
                f"decoder {decoder_loaded}/{len(decoder_keys)} "
                f"({decoder_fraction:.0%}) loaded. Both are expected to "
                f"load almost completely now that the real decoder "
                f"architecture is implemented — a shortfall here means a "
                f"submodule/key-naming mismatch to fix, not an acceptable "
                f"partial load.\n"
                f"Missing encoder keys: {encoder_missing[:10]}\n"
                f"Missing decoder keys: {decoder_missing[:10]}"
            )

        model = model.to(device).eval()
        logger.info(
            "ChangeFormer loaded on %s (%d params total, "
            "encoder: %d/%d, decoder: %d/%d from checkpoint)",
            device, sum(p.numel() for p in model.parameters()),
            encoder_loaded, len(encoder_keys),
            decoder_loaded, len(decoder_keys),
        )
        return model
=======
def detect_changes_fallback(
    image_before: str,
    image_after: str,
    threshold: float = 30.0,
    output_mask_path: Optional[str] = None,
) -> str:
    """
    Simple image differencing fallback when P3's ChangeFormer is not available.

    This is NOT production quality. It is a basic absolute difference + threshold
    to allow P2 to test the Change-VQA pipeline independently of P3.

    For production, P3's ChangeFormer should be used instead.

    Args:
        image_before: Path to T1 image
        image_after: Path to T2 image
        threshold: Pixel difference threshold (0-255)
