"""
SatQuery — Model Weight Downloader
=====================================
Downloads GeoChat-7B base model and LoRA adapter weights.

Usage:
    python scripts/download_models.py --base           # Download GeoChat-7B
    python scripts/download_models.py --adapter        # Download SatQuery-RS adapter
    python scripts/download_models.py --all            # Download everything
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GEOCHAT_MODEL_ID = "MBZUAI/geochat-7B"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"


def download_base_model(cache_dir: str | None = None) -> None:
    """Download GeoChat-7B base model from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("Install huggingface_hub: pip install huggingface-hub")
        sys.exit(1)

    token = os.getenv("HF_TOKEN", None)

    logger.info(f"Downloading GeoChat-7B from: {GEOCHAT_MODEL_ID}")
    logger.info("This will download ~14 GB of model weights...")

    try:
        path = snapshot_download(
            repo_id=GEOCHAT_MODEL_ID,
            local_dir=cache_dir,
            token=token,
            resume_download=True,
        )
        logger.info(f"✅ GeoChat-7B downloaded to: {path}")
    except Exception as e:
        logger.error(f"Failed to download: {e}")
        logger.info(
            "Make sure you:\n"
            "  1. Have a HuggingFace account\n"
            "  2. Accepted the model license at https://huggingface.co/MBZUAI/geochat-7B\n"
            "  3. Set HF_TOKEN in your .env file or environment\n"
            "  4. Run: huggingface-cli login"
        )
        sys.exit(1)


def download_vision_tower(cache_dir: str | None = None) -> None:
    """Download CLIP ViT-L/14 vision tower."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("Install huggingface_hub: pip install huggingface-hub")
        sys.exit(1)

    logger.info(f"Downloading CLIP vision tower: {CLIP_MODEL_ID}")

    try:
        path = snapshot_download(
            repo_id=CLIP_MODEL_ID,
            local_dir=cache_dir,
            resume_download=True,
        )
        logger.info(f"✅ CLIP vision tower downloaded to: {path}")
    except Exception as e:
        logger.error(f"Failed to download CLIP: {e}")
        sys.exit(1)


def download_adapter(hub_repo: str | None = None) -> None:
    """Download SatQuery-RS LoRA adapter from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("Install huggingface_hub: pip install huggingface-hub")
        sys.exit(1)

    if hub_repo is None:
        hub_repo = os.getenv("HF_HUB_REPO", "")

    if not hub_repo:
        logger.warning(
            "No adapter repo specified. Set HF_HUB_REPO in .env or use --repo flag.\n"
            "If you haven't fine-tuned yet, run the training pipeline first."
        )
        return

    adapter_dir = Path("models/checkpoints/satquery-rs-vlm")
    adapter_dir.mkdir(parents=True, exist_ok=True)

    token = os.getenv("HF_TOKEN", None)

    logger.info(f"Downloading SatQuery-RS adapter from: {hub_repo}")
    try:
        path = snapshot_download(
            repo_id=hub_repo,
            local_dir=str(adapter_dir),
            token=token,
            resume_download=True,
        )
        logger.info(f"✅ Adapter downloaded to: {path}")
    except Exception as e:
        logger.error(f"Failed to download adapter: {e}")


def check_status() -> None:
    """Check what's already downloaded."""
    logger.info("\n--- Model Download Status ---")

    # Check HF cache
    from pathlib import Path
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"

    geochat_cached = any(
        d.name.startswith("models--MBZUAI--geochat")
        for d in hf_cache.iterdir()
        if d.is_dir()
    ) if hf_cache.exists() else False

    adapter_exists = (
        Path("models/checkpoints/satquery-rs-vlm/adapter_config.json").exists()
    )

    logger.info(f"  GeoChat-7B base: {'✅ Downloaded' if geochat_cached else '❌ Not found'}")
    logger.info(f"  SatQuery-RS adapter: {'✅ Found' if adapter_exists else '❌ Not found (run training first)'}")
    logger.info("")


def main():
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument("--base", action="store_true", help="Download GeoChat-7B base model")
    parser.add_argument("--vision", action="store_true", help="Download CLIP vision tower")
    parser.add_argument("--adapter", action="store_true", help="Download SatQuery-RS adapter")
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument("--repo", type=str, default=None, help="HF Hub repo for adapter")
    parser.add_argument("--cache-dir", type=str, default=None, help="Custom cache directory")
    parser.add_argument("--status", action="store_true", help="Check download status")

    args = parser.parse_args()

    if args.status:
        check_status()
        return

    if args.all or args.base:
        download_base_model(args.cache_dir)

    if args.all or args.vision:
        download_vision_tower(args.cache_dir)

    if args.all or args.adapter:
        download_adapter(args.repo)

    if not any([args.all, args.base, args.vision, args.adapter, args.status]):
        parser.print_help()


if __name__ == "__main__":
    main()
