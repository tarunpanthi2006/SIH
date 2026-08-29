"""
SatQuery — Prithvi-EO-2.0 Inference Pipeline
===============================================

End-to-end: load 6-band multispectral HLS image → preprocess into
[B, C, T, H, W] format → extract features → SpecialistOutput.

Input bands (in order): Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2
Resolution: 30m (HLS product standard)
Tensor shape: [B, C, T, H, W] where T=1 for single-image inference
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from backend.tools.contracts import (
    Artifact,
    SpecialistOutput,
    TaskType,
    make_error,
    make_success,
)
from models.multispectral.model import PrithviEO2Model

logger = logging.getLogger(__name__)

MODEL_NAME = "Prithvi-EO-2.0"
DEFAULT_IMG_SIZE = 224  # Must be divisible by patch_size (16) — 224 = 14*16

# HLS band statistics for z-score normalization
# Source: Prithvi-EO-2.0 pretraining stats (NASA HLS V2)
# Order:  Blue    Green   Red     Narrow NIR  SWIR 1   SWIR 2
HLS_MEANS = [0.0334, 0.0575, 0.0898, 0.2241, 0.2312, 0.1601]
HLS_STDS  = [0.0357, 0.0475, 0.0763, 0.1118, 0.1062, 0.0963]


# ── Preprocessing ──────────────────────────────────────────────────────────

def preprocess_multispectral(
    image_path: str | Path,
    target_size: int = DEFAULT_IMG_SIZE,
    device: str = "cpu",
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Load and preprocess a 6-band HLS multispectral image.

    Parameters
    ----------
    image_path : path
        Path to a GeoTIFF with at least 6 spectral bands in the order:
        Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2.
        Values should be surface reflectance (HLS scale: 0–10000 range
        typical; the function normalises to [0,1] before z-scoring).
    target_size : int
        Spatial size to resize to. Must be a multiple of 16 (patch size).
    device : str
        "cuda" or "cpu".

    Returns
    -------
    tensor : Tensor [1, 6, 1, H, W]
        Ready for Prithvi-EO-2.0 forward pass.
        Shape: [B=1, C=6, T=1, target_size, target_size]
    original_size : (orig_h, orig_w)
        Native spatial resolution before resizing.
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Multispectral image not found: {p}")

    # ── Load raster ──
    try:
        import rasterio
        with rasterio.open(str(p)) as src:
            data = src.read().astype(np.float32)  # [C, H, W]
    except ImportError:
        # Fallback: PIL (for synthetic test images without rasterio)
        from PIL import Image
        img = Image.open(p)
        arr = np.array(img).astype(np.float32)
        if arr.ndim == 2:
            data = arr[np.newaxis, ...]           # [1, H, W]
        elif arr.ndim == 3:
            data = arr.transpose(2, 0, 1)         # [C, H, W]
        else:
            raise ValueError(f"Unexpected image shape: {arr.shape}")

    orig_h, orig_w = data.shape[1], data.shape[2]
    n_bands = data.shape[0]

    # ── Band selection / padding ──
    # We need exactly 6 bands in the HLS order.
    if n_bands > 6:
        data = data[:6]
        logger.warning("Image has %d bands; using first 6 as HLS bands.", n_bands)
    elif n_bands < 6:
        # Pad with zeros for missing bands — warn loudly.
        pad = np.zeros((6 - n_bands, orig_h, orig_w), dtype=np.float32)
        data = np.concatenate([data, pad], axis=0)
        logger.warning(
            "Image has only %d bands; padded to 6 with zeros. "
            "Results will be degraded — provide all 6 HLS bands.",
            n_bands,
        )

    # ── Reflectance scale normalisation ──
    # HLS V2 stores surface reflectance as integers 0–10000.
    # Divide to get physical reflectance [0, 1] before z-scoring.
    if data.max() > 10.0:
        data = data / 10000.0

    # Clip to [0, 1] (rare nodata values can go negative or above 1)
    data = np.clip(data, 0.0, 1.0)

    # ── Z-score normalisation ──
    means = np.array(HLS_MEANS, dtype=np.float32).reshape(6, 1, 1)
    stds  = np.array(HLS_STDS,  dtype=np.float32).reshape(6, 1, 1)
    data  = (data - means) / stds

    # ── To tensor: [1, 6, 1, H, W] (B=1, C=6, T=1, H, W) ──
    # Prithvi's 3D patch_embed conv has weight shape [out, 6, 1, 14, 14]
    # meaning PyTorch expects [B, C_in=6, D=T, H, W] — channels BEFORE time.
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(2)  # [1, 6, 1, H, W]

    # Resize spatial dims: F.interpolate doesn't support 5D directly,
    # so collapse B×C into one dim, resize, then restore.
    b, c, t, h, w = tensor.shape
    tensor_2d = tensor.view(b * c, t, h, w)  # [6, 1, H, W]
    tensor_2d = F.interpolate(
        tensor_2d, size=(target_size, target_size),
        mode="bilinear", align_corners=False,
    )
    tensor = tensor_2d.view(b, c, t, target_size, target_size)

    return tensor.to(device), (orig_h, orig_w)


# ── Model cache ────────────────────────────────────────────────────────────

_cached_model: PrithviEO2Model | None = None
_cached_device: str = ""
_cached_backbone: str = ""


def _get_model(
    backbone_name: str,
    local_dir: str | Path,
    device: str = "cuda",
) -> PrithviEO2Model:
    global _cached_model, _cached_device, _cached_backbone

    if (
        _cached_model is not None
        and _cached_device == device
        and _cached_backbone == backbone_name
    ):
        return _cached_model

    _cached_model = PrithviEO2Model.from_pretrained(
        backbone_name=backbone_name,
        local_dir=local_dir,
        device=device,
    )
    _cached_device = device
    _cached_backbone = backbone_name
    return _cached_model


# ── Main entry point ──────────────────────────────────────────────────────

def run_multispectral(
    image: str | Path,
    backbone_name: str = "prithvi_eo_v2_600",
    local_dir: str | Path = "checkpoints/prithvi",
    device: str = "cuda",
    output_dir: str | Path = "outputs/multispectral",
    query: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> SpecialistOutput:
    """
    Run multispectral feature extraction using Prithvi-EO-2.0.

    Parameters
    ----------
    image : path
        Path to a 6-band HLS GeoTIFF (Blue, Green, Red, NIR, SWIR1, SWIR2).
    backbone_name : str
        TerraTorch backbone name. Default "prithvi_eo_v2_600" (600M, no TL).
        Use "prithvi_eo_v2_600_tl" if you have timestamps + location to pass.
    local_dir : path
        Local checkpoint directory from download_models.py.
    device : str
        "cuda" or "cpu".
    output_dir : path
        Directory for output artifacts.
    query : str, optional
        User question (stored in metadata for downstream use).
    metadata : dict, optional
        Extra geospatial metadata (gsd, crs, bounds, timestamps, location…).

    Returns
    -------
    SpecialistOutput
        Contains extracted feature embeddings as an artifact and a summary
        answer. The embeddings can be passed to a downstream classification
        head (e.g. land cover, crop type) as defined by Person 1's workflow.
    """
    t0 = time.time()
    warnings_list: list[str] = []
    meta = dict(metadata or {})

    # ── 1. Validate input ──
    if not Path(image).exists():
        return make_error(TaskType.MULTISPECTRAL, MODEL_NAME,
                          f"Image not found: {image}")

    # ── 2. Preprocess ──
    try:
        if not torch.cuda.is_available() and device == "cuda":
            device = "cpu"
            warnings_list.append("CUDA not available, falling back to CPU.")

        tensor, orig_size = preprocess_multispectral(image, device=device)
    except Exception as exc:
        return make_error(TaskType.MULTISPECTRAL, MODEL_NAME,
                          f"Preprocessing failed: {exc}")

    # ── 3. Load model ──
    try:
        model = _get_model(backbone_name, local_dir, device)
    except Exception as exc:
        return make_error(TaskType.MULTISPECTRAL, MODEL_NAME,
                          f"Failed to load model: {exc}")

    # ── 4. Extract features ──
    try:
        # Pass optional temporal/location metadata if available
        timestamps = None
        location = None

        if "timestamps" in meta:
            # Expected: list of [[year, day_of_year]] for each time step
            try:
                ts_arr = np.array(meta["timestamps"], dtype=np.float32)
                timestamps = torch.from_numpy(ts_arr).unsqueeze(0).to(device)
            except Exception:
                warnings_list.append(
                    "Could not parse 'timestamps' from metadata; ignoring."
                )

        if "location" in meta:
            # Expected: [latitude, longitude]
            try:
                loc_arr = np.array(meta["location"], dtype=np.float32)
                location = torch.from_numpy(loc_arr).unsqueeze(0).to(device)
            except Exception:
                warnings_list.append(
                    "Could not parse 'location' from metadata; ignoring."
                )

        with torch.no_grad():
            features = model(tensor, timestamps=timestamps, location=location)

    except Exception as exc:
        return make_error(TaskType.MULTISPECTRAL, MODEL_NAME,
                          f"Inference failed: {exc}")

    # ── 5. Save artifacts ──
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_array = features.cpu().numpy()
    feat_path = out_dir / "multispectral_features.npy"
    np.save(str(feat_path), feat_array)

    artifacts = [
        Artifact(
            path=str(feat_path),
            description=(
                f"Prithvi-EO-2.0 feature embeddings, shape {list(feat_array.shape)}. "
                "Pass to a downstream segmentation or classification head."
            ),
        )
    ]

    # ── 6. Build answer ──
    if query:
        meta["user_query"] = query

    answer = (
        f"Successfully extracted multispectral features using {MODEL_NAME} "
        f"({backbone_name}). Feature tensor shape: {list(feat_array.shape)}. "
        "Downstream classification or segmentation head required for task-specific output."
    )

    elapsed = time.time() - t0

    return make_success(
        task=TaskType.MULTISPECTRAL,
        model=MODEL_NAME,
        answer=answer,
        confidence=0.9,
        artifacts=artifacts,
        metadata={
            "image_path": str(image),
            "backbone": backbone_name,
            "feature_shape": list(feat_array.shape),
            "input_shape": [1, 6, 1, DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE],
            "bands": ["Blue", "Green", "Red", "Narrow_NIR", "SWIR_1", "SWIR_2"],
            "native_resolution_m": 30,
            "original_size": list(orig_size),
            **meta,
        },
        warnings=warnings_list,
        inference_time_s=round(elapsed, 3),
    )
