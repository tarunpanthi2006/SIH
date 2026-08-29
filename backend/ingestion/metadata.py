"""
SatQuery — Metadata Extractor

Extracts structured ``ImageMetadata`` from an image file.
Uses rasterio for GeoTIFFs and Pillow for standard images.
Handles missing/incomplete metadata gracefully.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.api.schemas import ImageMetadata, Modality

logger = logging.getLogger(__name__)


def extract_metadata(path: str | Path) -> ImageMetadata:
    """
    Extract metadata from an image file.

    Parameters
    ----------
    path : str or Path
        Filesystem path to the image.

    Returns
    -------
    ImageMetadata
        Structured metadata.  Fields that cannot be determined are left
        as their schema defaults (``None`` / ``0`` / ``""``).
    """
    p = Path(path)
    if not p.exists():
        return ImageMetadata(path=str(p))

    suffix = p.suffix.lower()
    if suffix in {".tif", ".tiff", ".geotiff"}:
        return _extract_geotiff(p)
    return _extract_standard(p)


# ------------------------------------------------------------------ #
# GeoTIFF
# ------------------------------------------------------------------ #

def _extract_geotiff(path: Path) -> ImageMetadata:
    try:
        import rasterio  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("rasterio not installed — limited metadata for %s", path)
        return _extract_standard(path)

    try:
        with rasterio.open(path) as src:
            tags = src.tags()
            acq_date = _parse_acquisition_date(tags, path)
            sensor = tags.get("TIFFTAG_SOFTWARE") or tags.get("sensor")

            crs_str = str(src.crs) if src.crs else None
            bounds = list(src.bounds) if src.bounds else None
            resolution = tuple(src.res) if src.res else None

            modality = _guess_modality_from_bands(src.count, src.dtypes[0])

            return ImageMetadata(
                path=str(path),
                width=src.width,
                height=src.height,
                bands=src.count,
                dtype=str(src.dtypes[0]),
                crs=crs_str,
                bounds=bounds,
                resolution=resolution,
                acquisition_date=acq_date,
                sensor=sensor,
                modality=modality,
                extra=dict(tags),
            )
    except Exception as exc:
        logger.error("Metadata extraction failed for %s: %s", path, exc)
        return ImageMetadata(path=str(path))


# ------------------------------------------------------------------ #
# Standard (Pillow)
# ------------------------------------------------------------------ #

def _extract_standard(path: Path) -> ImageMetadata:
    try:
        from PIL import Image  # type: ignore[import-untyped]
    except ImportError:
        return ImageMetadata(path=str(path))

    try:
        img = Image.open(path)
        bands = len(img.getbands())
        return ImageMetadata(
            path=str(path),
            width=img.width,
            height=img.height,
            bands=bands,
            dtype="uint8",
            modality=Modality.OPTICAL if bands <= 4 else Modality.MULTISPECTRAL,
        )
    except Exception as exc:
        logger.error("Metadata extraction failed for %s: %s", path, exc)
        return ImageMetadata(path=str(path))


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

_DATE_PATTERNS = [
    r"\d{4}-\d{2}-\d{2}",       # 2024-03-15
    r"\d{4}_\d{2}_\d{2}",       # 2024_03_15
    r"\d{8}",                    # 20240315
]


def _parse_acquisition_date(
    tags: dict[str, Any], path: Path
) -> str | None:
    """Try to find an acquisition date from TIFF tags or the filename."""
    # Check common TIFF tags
    for key in ("TIFFTAG_DATETIME", "datetime", "acquisition_date", "DATE"):
        if key in tags and tags[key]:
            return str(tags[key])

    # Heuristic: try to parse a date from the filename
    stem = path.stem
    for pattern in _DATE_PATTERNS:
        match = re.search(pattern, stem)
        if match:
            raw = match.group(0).replace("_", "-")
            try:
                if len(raw) == 8 and "-" not in raw:
                    dt = datetime.strptime(raw, "%Y%m%d")
                else:
                    dt = datetime.strptime(raw, "%Y-%m-%d")
                return dt.date().isoformat()
            except ValueError:
                continue

    return None


def _guess_modality_from_bands(
    band_count: int, dtype: str
) -> Modality:
    """
    Best-effort modality guess from band count and data type.

    - 1 band + float → likely SAR (amplitude/intensity)
    - 1-4 bands + uint8 → likely optical RGB(A)
    - >4 bands → likely multispectral
    """
    if band_count == 1 and "float" in dtype:
        return Modality.SAR
    if band_count <= 4:
        return Modality.OPTICAL
    return Modality.MULTISPECTRAL
