"""
SatQuery — Image Loader

Loads GeoTIFF, TIFF, and standard images (PNG/JPEG) into a uniform
representation.  GeoTIFFs are opened with rasterio; everything else
falls back to Pillow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Supported file extensions
GEOTIFF_EXTENSIONS = {".tif", ".tiff", ".geotiff"}
STANDARD_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
ALL_EXTENSIONS = GEOTIFF_EXTENSIONS | STANDARD_EXTENSIONS


@dataclass
class LoadedImage:
    """Uniform wrapper around an opened image."""

    path: str
    array: np.ndarray                  # (bands, H, W) or (H, W, channels)
    metadata: dict[str, Any] = field(default_factory=dict)
    is_geotiff: bool = False


def load_image(path: str | Path) -> LoadedImage:
    """
    Open an image file and return a ``LoadedImage``.

    Parameters
    ----------
    path : str or Path
        Filesystem path to the image.

    Returns
    -------
    LoadedImage

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is unsupported.
    RuntimeError
        If the file cannot be read.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    suffix = p.suffix.lower()
    if suffix not in ALL_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{suffix}'. "
            f"Supported: {sorted(ALL_EXTENSIONS)}"
        )

    if suffix in GEOTIFF_EXTENSIONS:
        return _load_geotiff(p)
    return _load_standard(p)


# ------------------------------------------------------------------ #
# Private loaders
# ------------------------------------------------------------------ #

def _load_geotiff(path: Path) -> LoadedImage:
    """Load via rasterio (GeoTIFF / TIFF)."""
    try:
        import rasterio  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "rasterio not installed — falling back to Pillow for %s", path
        )
        return _load_standard(path)

    try:
        with rasterio.open(path) as src:
            array = src.read()  # (bands, H, W)
            meta: dict[str, Any] = {
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "dtype": str(src.dtypes[0]),
                "crs": str(src.crs) if src.crs else None,
                "bounds": list(src.bounds) if src.bounds else None,
                "transform": list(src.transform) if src.transform else None,
                "resolution": src.res if src.res else None,
                "tags": dict(src.tags()),
                "driver": src.driver,
            }
        return LoadedImage(
            path=str(path), array=array, metadata=meta, is_geotiff=True
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to read GeoTIFF {path}: {exc}") from exc


def _load_standard(path: Path) -> LoadedImage:
    """Load via Pillow (PNG / JPEG / etc.)."""
    try:
        from PIL import Image  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("Pillow is required but not installed.") from exc

    try:
        img = Image.open(path)
        img.load()
        array = np.asarray(img)
        # Normalise to (H, W, C) — single-channel gets an extra dim
        if array.ndim == 2:
            array = array[:, :, np.newaxis]

        meta: dict[str, Any] = {
            "width": img.width,
            "height": img.height,
            "bands": array.shape[2],
            "dtype": str(array.dtype),
            "mode": img.mode,
            "format": img.format,
        }
        return LoadedImage(
            path=str(path), array=array, metadata=meta, is_geotiff=False
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to read image {path}: {exc}") from exc
