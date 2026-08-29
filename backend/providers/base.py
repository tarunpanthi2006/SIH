"""
SatQuery — Data Provider Interface (Phase 2 Ready)

Defines the abstract ``DataProvider`` that future data sources
(Copernicus, GEE, Planetary Computer) will implement.

Phase 1 uses ``LocalUploadProvider`` — images come from local
filesystem paths provided by the user.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ImageReference:
    """A reference to an image that a provider can fetch."""

    id: str = ""
    source: str = ""
    path: str | None = None
    url: str | None = None
    modality: str = "unknown"
    acquisition_date: str | None = None
    bbox: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DataProvider(ABC):
    """
    Abstract base for image data sources.

    Phase 2 implementations:
    - ``CopernicusProvider``  → Copernicus Data Space
    - ``EarthEngineProvider`` → Google Earth Engine
    - ``PlanetaryComputerProvider`` → Microsoft Planetary Computer
    """

    name: str = "base"

    @abstractmethod
    async def search(
        self,
        bbox: list[float] | None = None,
        time_range: tuple[str, str] | None = None,
        modality: str | None = None,
        **kwargs: Any,
    ) -> list[ImageReference]:
        """
        Search for available images matching the criteria.

        Parameters
        ----------
        bbox : list[float], optional
            [left, bottom, right, top] geographic bounding box.
        time_range : tuple[str, str], optional
            (start_date, end_date) in ISO-8601.
        modality : str, optional
            Filter by modality (optical, sar, multispectral).

        Returns
        -------
        list[ImageReference]
        """
        ...

    @abstractmethod
    async def fetch(self, reference: ImageReference) -> Path:
        """
        Download / stage an image locally and return its path.

        Parameters
        ----------
        reference : ImageReference

        Returns
        -------
        Path
            Local filesystem path to the fetched image.
        """
        ...


# ================================================================== #
# Phase 1 implementation
# ================================================================== #

class LocalUploadProvider(DataProvider):
    """
    Phase 1 provider: images are already on the local filesystem.
    ``search()`` is a no-op; ``fetch()`` just returns the existing path.
    """

    name = "local_upload"

    async def search(
        self,
        bbox: list[float] | None = None,
        time_range: tuple[str, str] | None = None,
        modality: str | None = None,
        **kwargs: Any,
    ) -> list[ImageReference]:
        # Phase 1: no remote search — images are user-provided
        return []

    async def fetch(self, reference: ImageReference) -> Path:
        if reference.path is None:
            raise ValueError("LocalUploadProvider requires a local path.")
        p = Path(reference.path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        return p
