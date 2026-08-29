"""
SatQuery — Central Configuration

Uses pydantic-settings to read SATQUERY_* environment variables
with sensible defaults.  Import `get_settings()` anywhere.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    # -- Core --
    mock_mode: bool = True
    log_level: str = "INFO"

    # -- Directories --
    upload_dir: Path = Path("./uploads")
    artifacts_dir: Path = Path("./artifacts")

    # -- API Keys --
    gemini_api_key: str | None = Field(
        default=None, 
        validation_alias="GEMINI_API_KEY",
        description="API Key for Google Gemini"
    )

    # -- Limits --
    max_image_size_mb: int = 500

    model_config = {
        "env_prefix": "SATQUERY_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def ensure_dirs(self) -> None:
        """Create upload / artifact directories if they don't exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of application settings."""
    return Settings()
