"""
SatQuery — Multispectral Inference Pipeline
=============================================

End-to-end: load multispectral (HLS bands) → preprocess → predict → SpecialistOutput.
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
DEFAULT_IMG_SIZE = 224

# Means and stds for HLS bands (Blue, Green, Red, NIR, SWIR1, SWIR2)
# These are typical values for Prithvi, adjust based on exact model card if needed.
HLS_MEANS = [0.0334, 0.0575, 0.0898, 0.2241, 0.2312, 0.1601]
HLS_STDS = [0.0357, 0.0475, 0.0763, 0.1118, 0.1062, 0.0963]


def preprocess_multispectral(
    image_path: str | Path,
    target_size: int = DEFAULT_IMG_SIZE,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Load and preprocess multispectral HLS data.

    Expects a 6-band GeoTIFF.
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Multispectral image not found: {p}")

    try:
        import rasterio
        with rasterio.open(str(p)) as src:
            data = src.read()  # [C, H, W]
    except ImportError:
        # Fallback (e.g. for synthetic testing without rasterio)
        from PIL import Image
        img = Image.open(p)
        arr = np.array(img).astype(np.float32)
        # If it's a mock RGB image for testing, duplicate channels to make 6
        if arr.ndim == 3 and arr.shape[-1] == 3:
            data = np.concatenate([arr, arr], axis=-1).transpose(2, 0, 1)
        else:
            raise ValueError(f"Cannot load proper multispectral data without rasterio from {p}")

    data = data.astype(np.float32)

    # Basic normalization (min-max to [0,1] followed by z-score, depending on data scale)
    # Assuming data is in 10000 scale (surface reflectance)
    if data.max() > 10.0:
        data = data / 10000.0

    tensor = torch.from_numpy(data)  # [C, H, W]

    # Apply z-score normalization
    means = torch.tensor(HLS_MEANS).view(-1, 1, 1)
    stds = torch.tensor(HLS_STDS).view(-1, 1, 1)

    # Pad or slice bands if not exactly 6
    if tensor.shape[0] < 6:
        # Pad with zeros
        pad = torch.zeros((6 - tensor.shape[0], tensor.shape[1], tensor.shape[2]))
        tensor = torch.cat([tensor, pad], dim=0)
    elif tensor.shape[0] > 6:
        tensor = tensor[:6]

    tensor = (tensor - means) / stds

    tensor = tensor.unsqueeze(0)  # [1, C, H, W]
    tensor = F.interpolate(tensor, size=(target_size, target_size),
                            mode="bilinear", align_corners=False)

    return tensor.to(device)


_cached_model: PrithviEO2Model | None = None
_cached_device: str = ""
_cached_model_id: str = ""


def _get_model(hf_model_id: str, device: str = "cuda") -> PrithviEO2Model:
    global _cached_model, _cached_device, _cached_model_id
    if _cached_model is not None and _cached_device == device and _cached_model_id == hf_model_id:
        return _cached_model
    _cached_model = PrithviEO2Model.from_pretrained(hf_model_id, device=device)
    _cached_device = device
    _cached_model_id = hf_model_id
    return _cached_model


def run_multispectral(
    image: str | Path,
    hf_model_id: str = "ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
    device: str = "cuda",
    output_dir: str | Path = "outputs/multispectral",
    query: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> SpecialistOutput:
    """
    Run multispectral analysis using Prithvi-EO-2.0.
    """
    t0 = time.time()
    warnings_list: list[str] = []
    meta = dict(metadata or {})

    if not Path(image).exists():
        return make_error(TaskType.MULTISPECTRAL, MODEL_NAME, f"Image not found: {image}")

    try:
        if not torch.cuda.is_available() and device == "cuda":
            device = "cpu"
            warnings_list.append("CUDA not available, falling back to CPU.")

        tensor = preprocess_multispectral(image, device=device)
    except Exception as exc:
        return make_error(TaskType.MULTISPECTRAL, MODEL_NAME, f"Preprocessing failed: {exc}")

    try:
        model = _get_model(hf_model_id, device)
    except Exception as exc:
        return make_error(TaskType.MULTISPECTRAL, MODEL_NAME, f"Failed to load model: {exc}")

    try:
        with torch.no_grad():
            features = model(tensor)
    except Exception as exc:
        return make_error(TaskType.MULTISPECTRAL, MODEL_NAME, f"Inference failed: {exc}")

    # Output generation
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    feat_path = out_dir / "multispectral_features.npy"
    feat_array = features.cpu().numpy()
    np.save(str(feat_path), feat_array)

    artifacts = [
        Artifact(
            path=str(feat_path),
            description="Extracted multispectral feature embeddings [B, N, D]",
        )
    ]

    answer = "Successfully extracted multispectral features using Prithvi-EO-2.0."
    if query:
        meta["user_query"] = query

    elapsed = time.time() - t0

    return make_success(
        task=TaskType.MULTISPECTRAL,
        model=MODEL_NAME,
        answer=answer,
        confidence=0.9,  # High confidence for feature extraction success
        artifacts=artifacts,
        metadata={
            "image_path": str(image),
            "model_id": hf_model_id,
            "feature_shape": list(feat_array.shape),
            **meta,
        },
        warnings=warnings_list,
        inference_time_s=round(elapsed, 3),
    )
