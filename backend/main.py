"""
SatQuery — Application Entry Point

Creates the FastAPI app, registers tools, and mounts the API router.
Run with:  uvicorn backend.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.agent.registry import get_registry, reset_registry
from backend.api.routes import router as api_router
from backend.config import get_settings

# -- Tool imports (mock or real) --
from backend.tools.caption import MockCaptionTool
from backend.tools.change import MockChangeDetectionTool, MockChangeVQATool
from backend.tools.grounding import MockGroundingTool
from backend.tools.multispectral import MockMultispectralTool
from backend.tools.optical_sar import MockOpticalSARTool
from backend.tools.vqa import MockVQATool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown logic."""
    settings = get_settings()
    settings.ensure_dirs()
    _register_tools(settings.mock_mode)
    logger = logging.getLogger("satquery")
    logger.info("SatQuery backend started (mock_mode=%s)", settings.mock_mode)
    yield
    # Shutdown
    reset_registry()


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()

    # -- Logging --
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    # -- FastAPI --
    app = FastAPI(
        title="SatQuery",
        description=(
            "Agentic multimodal remote-sensing intelligence system. "
            "SIH 2026 — Problem Statement 167."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # -- CORS (permissive for dev; tighten for production) --
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Router --
    app.include_router(api_router)

    return app


def _register_tools(mock_mode: bool) -> None:
    """
    Register all tools in the global registry.

    When ``mock_mode`` is True, register mock implementations.
    When False, register real implementations (Person 2 / 3 will
    provide these — for now they fall back to mocks).
    """
    registry = get_registry()

    if mock_mode:
        # -- Mock tools --
        registry.register(MockVQATool())
        registry.register(MockCaptionTool())
        registry.register(MockGroundingTool())
        registry.register(MockChangeDetectionTool())
        registry.register(MockChangeVQATool())
        registry.register(MockOpticalSARTool())
        registry.register(MockMultispectralTool())
    else:
        # -- Real tools (to be implemented by Person 2 / 3) --
        # For now, fall back to mocks to keep the system runnable.
        registry.register(MockVQATool())
        registry.register(MockCaptionTool())
        registry.register(MockGroundingTool())
        registry.register(MockChangeDetectionTool())
        registry.register(MockChangeVQATool())
        registry.register(MockOpticalSARTool())
        registry.register(MockMultispectralTool())
        logging.getLogger("satquery").warning(
            "MOCK_MODE is off but real tools are not yet integrated — "
            "falling back to mock implementations."
        )


# -- Module-level app instance for uvicorn --
app = create_app()
