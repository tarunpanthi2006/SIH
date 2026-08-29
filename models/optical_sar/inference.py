"""
SatQuery — SkySense++ Inference Pipeline
==========================================

End-to-end: load optical & SAR → validate co-registration → preprocess
(proper SAR dB handling) → cross-modal fusion → land-cover classification →
SpecialistOutput.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backend.tools.contracts import (
    Artifact,
    EvidenceType,
    SpatialEvidence,
    SpecialistOutput,
    TaskType,
    make_error,
    make_success,
)
from models.optical_sar.model import SkySensePPModel

logger = logging.getLogger(__name__)

MODEL_NAME = "SkySense++"
DEFAULT_IMG_SIZE = 224
LAND_COVER_CLASSES = [
    "water", "built_up", "vegetation", "bare_soil",
    "agriculture", "wetland", "snow_ice", "cloud", "other",
]


# ── SAR preprocessing ─────────────────────────────────────────────────────

def preprocess_sar(
    image_path: str | Path,
    target_size: int = DEFAULT_IMG_SIZE,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Load and preprocess a SAR image for SkySense++.

    SAR pixel values represent radar backscatter.  We do NOT treat them
    as ordinary RGB.  The pipeline:

    1. Load the image (expects 1-band or 2-band for VV/VH)
    2. If RGB TIFF with SAR data encoded, extract bands
    3. Convert linear power to dB: dB = 10 * log10(linear + eps)
    4. Clip to reasonable dB range (e.g. -30 to +5 dB)
    5. Normalise to [-1, 1] for the model

    Parameters
    ----------
    image_path : path
        Path to SAR image (GeoTIFF preferred, PNG/JPG accepted with warning).
    target_size : int
        Target spatial dimension.
    device : str
        Target device.

    Returns
    -------
    Tensor [1, 2, H, W]  (VV, VH channels) normalised to ~[-1, 1]
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"SAR image not found: {p}")

    warnings: list[str] = []

    # Try rasterio first for proper geospatial handling
    try:
        import rasterio
        with rasterio.open(str(p)) as src:
            data = src.read()  # [C, H, W]
            # If single-band, duplicate for VV/VH
            if data.shape[0] == 1:
                data = np.concatenate([data, data], axis=0)
                warnings.append("Single-band SAR image; duplicated for VV/VH channels.")
            elif data.shape[0] >= 2:
                data = data[:2]  # Take first 2 bands as VV, VH
            elif data.shape[0] == 3:
                # Might be RGB-encoded SAR; use R and G as VV, VH
                data = data[:2]
                warnings.append(
                    "3-band SAR image detected; using bands 0,1 as VV/VH."
                )
    except ImportError:
        # Fallback to PIL
        img = Image.open(p)
        arr = np.array(img).astype(np.float32)
        if arr.ndim == 2:
            data = np.stack([arr, arr], axis=0)
            warnings.append("Loaded SAR as grayscale via PIL; duplicated for VV/VH.")
        elif arr.ndim == 3:
            data = arr[:, :, :2].transpose(2, 0, 1)
        else:
            raise ValueError(f"Unexpected SAR image shape: {arr.shape}")

    data = data.astype(np.float32)

    # ── dB conversion ──
    # Check if data looks like linear power (positive, small values)
    # or already in dB (can be negative)
    data_min = float(data.min())
    data_max = float(data.max())

    if data_min >= 0 and data_max > 0:
        # Likely linear power → convert to dB
        eps = 1e-10
        data = 10.0 * np.log10(data + eps)
        logger.debug("SAR: converted linear power to dB (range was [%.2f, %.2f])",
                      data_min, data_max)
    else:
        logger.debug("SAR: data appears to already be in dB (range [%.2f, %.2f])",
                      data_min, data_max)

    # ── Clip to reasonable dB range ──
    DB_MIN, DB_MAX = -30.0, 5.0
    data = np.clip(data, DB_MIN, DB_MAX)

    # ── Normalise to [-1, 1] ──
    data = 2.0 * (data - DB_MIN) / (DB_MAX - DB_MIN) - 1.0

    # ── To tensor and resize ──
    tensor = torch.from_numpy(data).unsqueeze(0).float()  # [1, 2, H, W]
    tensor = F.interpolate(tensor, size=(target_size, target_size),
                            mode="bilinear", align_corners=False)

    return tensor.to(device)


# ── Optical preprocessing ─────────────────────────────────────────────────

def preprocess_optical(
    image_path: str | Path,
    target_size: int = DEFAULT_IMG_SIZE,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Load and preprocess an optical image for SkySense++.

    Parameters
    ----------
    image_path : path
        Path to optical image (RGB).
    target_size : int
        Target spatial dimension.
    device : str
        Target device.

    Returns
    -------
    Tensor [1, 3, H, W]  normalised to [0, 1]
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Optical image not found: {p}")

    img = Image.open(p).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    tensor = F.interpolate(tensor, size=(target_size, target_size),
                            mode="bilinear", align_corners=False)
    return tensor.to(device)


# ── Co-registration validation ─────────────────────────────────────────────

def validate_coregistration(
    optical_meta: dict[str, Any] | None,
    sar_meta: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    """
    Check whether optical and SAR images are co-registered.

    Checks (when metadata is available):
    1. Same CRS
    2. Geographic bounding boxes overlap
    3. Compatible spatial resolution

    Returns
    -------
    is_valid : bool
    warnings : list[str]
    """
    warnings: list[str] = []

    if not optical_meta or not sar_meta:
        warnings.append(
            "No geospatial metadata provided for co-registration check. "
            "Assuming images are already co-registered."
        )
        return True, warnings

    # CRS check
    opt_crs = optical_meta.get("crs")
    sar_crs = sar_meta.get("crs")
    if opt_crs and sar_crs and opt_crs != sar_crs:
        warnings.append(
            f"CRS mismatch: optical={opt_crs}, SAR={sar_crs}. "
            "Images may not be properly co-registered."
        )
        # Not a hard failure — model may still work on roughly aligned data
        return True, warnings

    # Bounding box overlap
    opt_bbox = optical_meta.get("bounds")
    sar_bbox = sar_meta.get("bounds")
    if opt_bbox and sar_bbox:
        # bounds = [left, bottom, right, top]
        overlap = (
            opt_bbox[0] < sar_bbox[2] and opt_bbox[2] > sar_bbox[0] and
            opt_bbox[1] < sar_bbox[3] and opt_bbox[3] > sar_bbox[1]
        )
        if not overlap:
            return False, [
                "No geographic overlap between optical and SAR images. "
                "They cover different areas and cannot be analysed jointly."
            ]

    # Resolution check
    opt_res = optical_meta.get("gsd")
    sar_res = sar_meta.get("gsd")
    if opt_res and sar_res:
        ratio = max(opt_res, sar_res) / min(opt_res, sar_res)
        if ratio > 10:
            warnings.append(
                f"Large resolution difference: optical={opt_res}m, SAR={sar_res}m "
                f"(ratio {ratio:.1f}x). Results may be degraded."
            )

    return True, warnings


# ── Classification post-processing ─────────────────────────────────────────

def _decode_scene_prediction(
    logits: torch.Tensor,
) -> tuple[dict[str, float], str]:
    """
    Decode scene-level logits to class probabilities.

    Returns
    -------
    probs_dict : dict mapping class name → probability
    top_class  : name of the highest-probability class
    """
    probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
    probs_dict = {
        cls_name: round(float(p), 4)
        for cls_name, p in zip(LAND_COVER_CLASSES, probs)
    }
    top_idx = int(np.argmax(probs))
    return probs_dict, LAND_COVER_CLASSES[top_idx]


def _decode_pixel_prediction(
    logits: torch.Tensor,
    original_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Decode per-pixel logits to classification map + area statistics.

    Returns
    -------
    class_map : ndarray [H, W] uint8 (class indices)
    area_fractions : dict mapping class name → fraction of total area
    """
    # [B, C, H', W'] → argmax → [H', W']
    if original_size:
        logits = F.interpolate(
            logits.float(),
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )
    class_map = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

    total = class_map.size
    fractions = {}
    for i, cls_name in enumerate(LAND_COVER_CLASSES):
        count = int((class_map == i).sum())
        fractions[cls_name] = round(count / total, 4) if total > 0 else 0.0

    return class_map, fractions


# ── Main entry point ──────────────────────────────────────────────────────

_cached_model: SkySensePPModel | None = None
_cached_device: str = ""
_cached_checkpoint: str = ""


def _get_model(
    checkpoint_path: str | Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> SkySensePPModel:
    global _cached_model, _cached_device, _cached_checkpoint
    cp = str(checkpoint_path)
    if _cached_model is not None and _cached_device == device and _cached_checkpoint == cp:
        return _cached_model
    _cached_model = SkySensePPModel.from_pretrained(cp, device=device, dtype=dtype)
    _cached_device = device
    _cached_checkpoint = cp
    return _cached_model


def run_optical_sar(
    optical: str | Path,
    sar: str | Path,
    checkpoint_path: str | Path = "checkpoints/skysensepp/",
    device: str = "cuda",
    output_dir: str | Path = "outputs/optical_sar",
    query: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> SpecialistOutput:
    """
    Run cross-modal optical + SAR analysis using SkySense++.

    This performs TRUE cross-modal fusion — the model jointly processes
    complementary information from optical (spectral) and SAR (structural)
    modalities, NOT independent analysis concatenated together.

    Parameters
    ----------
    optical : path
        Path to the optical image (RGB).
    sar : path
        Path to the SAR image (VV/VH or single-band backscatter).
    checkpoint_path : path
        SkySense++ checkpoint path.
    device : str
        'cuda' or 'cpu'.
    output_dir : path
        Directory for output artifacts.
    query : str, optional
        User question for context.
    metadata : dict, optional
        Geospatial metadata for both images.  Can contain sub-dicts
        'optical' and 'sar' with per-modality metadata.

    Returns
    -------
    SpecialistOutput
    """
    t0 = time.time()
    warnings_list: list[str] = []
    meta = dict(metadata or {})

    # ── 1. Validate inputs exist ──
    if not Path(optical).exists():
        return make_error(TaskType.OPTICAL_SAR, MODEL_NAME,
                          f"Optical image not found: {optical}")
    if not Path(sar).exists():
        return make_error(TaskType.OPTICAL_SAR, MODEL_NAME,
                          f"SAR image not found: {sar}")

    # ── 2. Co-registration check ──
    opt_meta = meta.get("optical", {})
    sar_meta_dict = meta.get("sar", {})
    is_valid, coreg_warnings = validate_coregistration(opt_meta, sar_meta_dict)
    warnings_list.extend(coreg_warnings)
    if not is_valid:
        return make_error(
            TaskType.OPTICAL_SAR, MODEL_NAME,
            "Co-registration validation failed: " + "; ".join(coreg_warnings),
        )

    # ── 3. Preprocess ──
    try:
        if not torch.cuda.is_available() and device == "cuda":
            device = "cpu"
            warnings_list.append("CUDA not available, falling back to CPU.")

        optical_tensor = preprocess_optical(optical, device=device)
        sar_tensor = preprocess_sar(sar, device=device)
    except Exception as exc:
        return make_error(TaskType.OPTICAL_SAR, MODEL_NAME,
                          f"Preprocessing failed: {exc}")

    # ── 4. Load model ──
    try:
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = _get_model(checkpoint_path, device, dtype)
        optical_tensor = optical_tensor.to(dtype=dtype)
        sar_tensor = sar_tensor.to(dtype=dtype)
    except Exception as exc:
        return make_error(TaskType.OPTICAL_SAR, MODEL_NAME,
                          f"Failed to load model: {exc}")

    # ── 5. Cross-modal inference ──
    try:
        with torch.no_grad():
            output = model(optical=optical_tensor, sar=sar_tensor, task="both")
    except Exception as exc:
        return make_error(TaskType.OPTICAL_SAR, MODEL_NAME,
                          f"Inference failed: {exc}")

    # ── 6. Decode results ──
    spatial_evidence = []
    statistics: dict[str, Any] = {}
    artifacts_list: list[Artifact] = []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scene-level classification
    if "scene_logits" in output:
        probs_dict, top_class = _decode_scene_prediction(output["scene_logits"])
        statistics["scene_classification"] = probs_dict
        statistics["dominant_class"] = top_class

    # Per-pixel classification map
    if "pixel_logits" in output:
        class_map, area_fractions = _decode_pixel_prediction(output["pixel_logits"])
        statistics["land_cover_fractions"] = area_fractions

        # Save classification map
        map_path = out_dir / "land_cover_map.png"
        # Colour-coded map
        colour_map = _colorise_class_map(class_map)
        Image.fromarray(colour_map).save(map_path)

        spatial_evidence.append(SpatialEvidence(
            type=EvidenceType.CLASSIFICATION_MAP,
            path=str(map_path),
            description="Per-pixel land-cover classification map from cross-modal analysis",
        ))
        artifacts_list.append(Artifact(
            path=str(map_path),
            description="Colour-coded land-cover classification map",
            mime_type="image/png",
        ))

        # Save raw class indices
        raw_path = out_dir / "land_cover_raw.npy"
        np.save(str(raw_path), class_map)
        artifacts_list.append(Artifact(
            path=str(raw_path),
            description="Raw class-index map (uint8 numpy array)",
        ))

    # ── 7. Build answer ──
    dominant = statistics.get("dominant_class", "unknown")
    fracs = statistics.get("land_cover_fractions", {})

    # Build a human-readable summary
    significant = {k: v for k, v in fracs.items() if v >= 0.05}
    parts = [f"{k} ({v:.0%})" for k, v in
             sorted(significant.items(), key=lambda x: x[1], reverse=True)]

    if parts:
        answer = (
            f"Cross-modal optical+SAR analysis identifies: {', '.join(parts)}. "
            f"Dominant land cover: {dominant}."
        )
    else:
        answer = (
            f"Cross-modal analysis complete. Dominant scene class: {dominant}."
        )

    confidence = float(max(fracs.values())) if fracs else 0.5

    if query:
        meta["user_query"] = query

    elapsed = time.time() - t0

    return make_success(
        task=TaskType.OPTICAL_SAR,
        model=MODEL_NAME,
        answer=answer,
        confidence=round(min(confidence, 0.95), 3),
        spatial_evidence=spatial_evidence,
        statistics=statistics,
        artifacts=artifacts_list,
        metadata={
            "optical_path": str(optical),
            "sar_path": str(sar),
            "sar_preprocessing": "dB_normalised",
            "fusion_type": "cross_modal_transformer",
            "classes": LAND_COVER_CLASSES,
            **meta,
        },
        warnings=warnings_list,
        inference_time_s=round(elapsed, 3),
    )


# ── Helpers ────────────────────────────────────────────────────────────────

# Distinct colours for each land-cover class
_CLASS_COLOURS = np.array([
    [0, 0, 200],      # water — blue
    [200, 0, 0],      # built_up — red
    [0, 180, 0],      # vegetation — green
    [180, 140, 80],   # bare_soil — tan
    [255, 255, 0],    # agriculture — yellow
    [0, 200, 200],    # wetland — cyan
    [255, 255, 255],  # snow_ice — white
    [180, 180, 180],  # cloud — grey
    [100, 100, 100],  # other — dark grey
], dtype=np.uint8)


def _colorise_class_map(class_map: np.ndarray) -> np.ndarray:
    """Convert class-index map [H,W] to RGB colour image [H,W,3]."""
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(_CLASS_COLOURS.shape[0]):
        rgb[class_map == i] = _CLASS_COLOURS[i]
    return rgb
