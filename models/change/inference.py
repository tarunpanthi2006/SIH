SatQuery — ChangeFormer Inference Pipeline
============================================

End-to-end: load images → preprocess → tile → predict → postprocess →
change mask + statistics → SpecialistOutput.
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
from models.change.model import ChangeFormerModel

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

DEFAULT_INPUT_SIZE = 256
DEFAULT_THRESHOLD = 0.5
TILE_OVERLAP = 32  # pixels of overlap when tiling large images
MODEL_NAME = "ChangeFormer"


# ── Image loading & validation ─────────────────────────────────────────────

def _load_image(path: str | Path) -> np.ndarray:
    """Load an image as RGB uint8 numpy array [H, W, 3]."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    img = Image.open(p).convert("RGB")
    return np.array(img)


def _validate_pair(img_a: np.ndarray, img_b: np.ndarray) -> list[str]:
    """Check that the pair is compatible.  Returns list of warnings."""
    warnings: list[str] = []
    if img_a.shape != img_b.shape:
        raise ValueError(
            f"Image dimensions do not match: {img_a.shape} vs {img_b.shape}.  "
            "Bi-temporal pairs must be co-registered with identical dimensions."
        )
    if img_a.ndim != 3 or img_a.shape[2] != 3:
        raise ValueError(f"Expected RGB images, got shape {img_a.shape}")
    h, w = img_a.shape[:2]
    if h < 32 or w < 32:
        raise ValueError(f"Image too small ({h}×{w}), minimum is 32×32")
    if h != w:
        warnings.append(f"Non-square image ({h}×{w}); will be padded.")
    return warnings


# ── Preprocessing ──────────────────────────────────────────────────────────

def _to_tensor(img: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """[H,W,3] uint8 → [1,3,H,W] float32 normalised to [0,1]."""
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return t.to(device)


def _pad_to_multiple(tensor: torch.Tensor, multiple: int = 32) -> tuple[torch.Tensor, tuple]:
    """Pad spatial dims to nearest multiple; return padded tensor + original size."""
    _, _, h, w = tensor.shape
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    if new_h == h and new_w == w:
        return tensor, (h, w)
    padded = F.pad(tensor, (0, new_w - w, 0, new_h - h), mode="reflect")
    return padded, (h, w)


# ── Tiled inference ────────────────────────────────────────────────────────

def _tiled_predict(
    model: ChangeFormerModel,
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    tile_size: int = DEFAULT_INPUT_SIZE,
    overlap: int = TILE_OVERLAP,
) -> torch.Tensor:
    """
    Run inference with overlapping tiles for large images.

    Returns probability map [1, H, W] of the change class.
    """
    _, _, H, W = tensor_a.shape
    stride = tile_size - overlap

    # If image fits in one tile, run directly
    if H <= tile_size and W <= tile_size:
        a_resized = F.interpolate(tensor_a, size=(tile_size, tile_size),
                                   mode="bilinear", align_corners=False)
        b_resized = F.interpolate(tensor_b, size=(tile_size, tile_size),
                                   mode="bilinear", align_corners=False)
        with torch.no_grad():
            logits = model(a_resized, b_resized)
        probs = F.softmax(logits, dim=1)[:, 1:2, :, :]  # change class
        probs = F.interpolate(probs, size=(H, W),
                              mode="bilinear", align_corners=False)
        return probs.squeeze(1)

    # Tiled inference with weight accumulation
    prob_sum = torch.zeros(1, 1, H, W, device=tensor_a.device)
    weight_sum = torch.zeros(1, 1, H, W, device=tensor_a.device)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            tile_a = tensor_a[:, :, y_start:y_end, x_start:x_end]
            tile_b = tensor_b[:, :, y_start:y_end, x_start:x_end]

            # Resize tile if it's not exactly tile_size
            th, tw = tile_a.shape[2], tile_a.shape[3]
            if th != tile_size or tw != tile_size:
                tile_a = F.interpolate(tile_a, (tile_size, tile_size),
                                        mode="bilinear", align_corners=False)
                tile_b = F.interpolate(tile_b, (tile_size, tile_size),
                                        mode="bilinear", align_corners=False)

            with torch.no_grad():
                logits = model(tile_a, tile_b)
            probs = F.softmax(logits, dim=1)[:, 1:2, :, :]

            # Resize back to tile dimensions
            probs = F.interpolate(probs, (y_end - y_start, x_end - x_start),
                                  mode="bilinear", align_corners=False)

            prob_sum[:, :, y_start:y_end, x_start:x_end] += probs
            weight_sum[:, :, y_start:y_end, x_start:x_end] += 1.0

    result = prob_sum / weight_sum.clamp(min=1.0)
    return result.squeeze(1)  # [1, H, W]


# ── Post-processing ───────────────────────────────────────────────────────

def _postprocess(
    prob_map: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert probability map to binary mask and probability array.

    Returns
    -------
    mask : ndarray [H, W] uint8 (0 or 255)
    prob : ndarray [H, W] float32
    """
    prob = prob_map.squeeze().cpu().numpy().astype(np.float32)
    mask = (prob >= threshold).astype(np.uint8) * 255
    return mask, prob


def _compute_statistics(
    mask: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute change statistics from the binary mask.

    If metadata contains 'gsd' (ground sampling distance in metres),
    computes area in m².  Otherwise only pixel-level stats.
    """
    total_pixels = int(mask.size)
    changed_pixels = int((mask > 0).sum())
    changed_fraction = changed_pixels / total_pixels if total_pixels > 0 else 0.0

    stats: dict[str, Any] = {
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
        "changed_fraction": round(changed_fraction, 6),
    }

    # Attempt area calculation if GSD available
    if metadata and "gsd" in metadata:
        try:
            gsd = float(metadata["gsd"])
            pixel_area_m2 = gsd * gsd
            stats["changed_area_m2"] = round(changed_pixels * pixel_area_m2, 2)
            stats["total_area_m2"] = round(total_pixels * pixel_area_m2, 2)
            stats["gsd_m"] = gsd
        except (ValueError, TypeError):
            stats["area_note"] = "GSD metadata present but not a valid number"
    else:
        stats["area_note"] = (
            "Area in real-world units unavailable — "
            "no GSD / CRS metadata provided."
        )

    return stats


def _find_change_regions(mask: np.ndarray) -> list[dict[str, Any]]:
    """
    Find connected change regions and compute their bounding boxes.

    Uses scipy for labelling if available, otherwise skips.
    """
    try:
        from scipy import ndimage
    except ImportError:
        return []

    labelled, num_features = ndimage.label(mask > 0)
    regions = []
    for i in range(1, num_features + 1):
        ys, xs = np.where(labelled == i)
        area = int(len(ys))
        if area < 4:  # skip noise
            continue
        regions.append({
            "id": i,
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "pixel_area": area,
        })

    # Sort by area descending
    regions.sort(key=lambda r: r["pixel_area"], reverse=True)
    return regions


# ── Main entry point ──────────────────────────────────────────────────────

_cached_model: ChangeFormerModel | None = None
_cached_device: str = ""
_cached_checkpoint: str = ""


def _get_model(
    checkpoint_path: str | Path,
    device: str = "cuda",
) -> ChangeFormerModel:
    """Get or create a cached model instance."""
    global _cached_model, _cached_device, _cached_checkpoint
    cp = str(checkpoint_path)
    if _cached_model is not None and _cached_device == device and _cached_checkpoint == cp:
        return _cached_model

    _cached_model = ChangeFormerModel.from_pretrained(cp, device=device)
    _cached_device = device
    _cached_checkpoint = cp
    return _cached_model


def run_change(
    image_a: str | Path,
    image_b: str | Path,
    checkpoint_path: str | Path = "checkpoints/changeformer/ChangeFormer_LEVIR.pth",
    device: str = "cuda",
    threshold: float = DEFAULT_THRESHOLD,
    output_dir: str | Path = "outputs/change",
    metadata: dict[str, Any] | None = None,
) -> SpecialistOutput:
    """
    Run bi-temporal change detection on a co-registered image pair.

    Parameters
    ----------
    image_a : path
        Path to the first (earlier) image.
    image_b : path
        Path to the second (later) image.
    checkpoint_path : path
        Path to the ChangeFormer .pth checkpoint.
    device : str
        'cuda' or 'cpu'.
    threshold : float
        Probability threshold for binarising the change mask.
    output_dir : path
        Directory for output artifacts (mask PNG, probability map).
    metadata : dict, optional
        Geospatial metadata.  If it contains 'gsd' (ground sampling
        distance in metres), area statistics are computed.

    Returns
    -------
    SpecialistOutput
        Structured output following the SatQuery contract.
    """
    t0 = time.time()
    warnings_list: list[str] = []

    # ── 1. Load & validate ──
    try:
        img_a = _load_image(image_a)
        img_b = _load_image(image_b)
        w = _validate_pair(img_a, img_b)
        warnings_list.extend(w)
    except (FileNotFoundError, ValueError, OSError) as exc:
        return make_error(TaskType.CHANGE_DETECTION, MODEL_NAME, str(exc))

    # ── 2. Load model ──
    try:
        if not torch.cuda.is_available() and device == "cuda":
            device = "cpu"
            warnings_list.append("CUDA not available, falling back to CPU.")
        model = _get_model(checkpoint_path, device)
    except Exception as exc:
        return make_error(
            TaskType.CHANGE_DETECTION, MODEL_NAME,
            f"Failed to load model: {exc}",
        )

    # ── 3. Preprocess ──
    tensor_a = _to_tensor(img_a, device)
    tensor_b = _to_tensor(img_b, device)
    tensor_a, orig_size = _pad_to_multiple(tensor_a)
    tensor_b, _ = _pad_to_multiple(tensor_b)

    # ── 4. Inference ──
    prob_map = _tiled_predict(model, tensor_a, tensor_b)

    # Crop back to original size
    h, w = orig_size
    prob_map = prob_map[:, :h, :w]

    # ── 5. Post-process ──
    mask, prob_arr = _postprocess(prob_map, threshold)
    stats = _compute_statistics(mask, metadata)
    regions = _find_change_regions(mask)
    if regions:
        stats["num_change_regions"] = len(regions)
        stats["largest_region_pixels"] = regions[0]["pixel_area"]
        stats["change_regions"] = regions[:20]  # top 20

    # ── 6. Save artifacts ──
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_path = out_dir / "change_mask.png"
    prob_path = out_dir / "change_prob.npy"
    Image.fromarray(mask).save(mask_path)
    np.save(str(prob_path), prob_arr)

    spatial_evidence = [
        SpatialEvidence(
            type=EvidenceType.MASK,
            path=str(mask_path),
            description="Binary change mask (white = changed, black = unchanged)",
        ),
    ]

    artifacts = [
        Artifact(
            path=str(prob_path),
            description="Change probability map (float32 numpy array)",
            mime_type="application/octet-stream",
        ),
    ]

    # ── 7. Build answer ──
    frac = stats["changed_fraction"]
    if frac < 0.001:
        answer = "No significant change detected between the two images."
        confidence = 1.0 - frac
    elif frac < 0.05:
        answer = (
            f"Minor change detected affecting {frac:.1%} of the image area. "
            f"{stats.get('num_change_regions', 0)} distinct change region(s) identified."
        )
        confidence = min(0.85, frac * 10 + 0.5)
    else:
        answer = (
            f"Significant change detected affecting {frac:.1%} of the image area. "
            f"{stats.get('num_change_regions', 0)} distinct change region(s) identified."
        )
        confidence = min(0.95, 0.7 + frac)

    elapsed = time.time() - t0

    return make_success(
        task=TaskType.CHANGE_DETECTION,
        model=MODEL_NAME,
        answer=answer,
        confidence=round(confidence, 3),
        spatial_evidence=spatial_evidence,
        statistics=stats,
        artifacts=artifacts,
        metadata={
            "image_a": str(image_a),
            "image_b": str(image_b),
            "threshold": threshold,
            "image_size": list(img_a.shape[:2]),
            **(metadata or {}),
        },
        warnings=warnings_list,
        inference_time_s=round(elapsed, 3),
    )
