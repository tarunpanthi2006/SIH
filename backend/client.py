"""
SatQuery-RS — Client Library
================================
Simple client for calling the SatQuery-RS model server from
Person 1's agent or any other code.

Works with both local and cloud-hosted servers.

Usage:
    from backend.client import SatQueryClient

    client = SatQueryClient("http://localhost:8100")
    # or: client = SatQueryClient("http://cloud-gpu-ip:8100")

    result = client.vqa("satellite.png", "What land cover is present?")
    result = client.caption("satellite.png")
    result = client.grounding("satellite.png", "water body")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


class SatQueryClient:
    """
    HTTP client for the SatQuery-RS model server.

    Person 1's agent uses this to call Person 2's VLM tools
    without importing the model directly.
    """

    def __init__(self, base_url: str = "http://localhost:8100", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        """Check if the server is healthy and model is loaded."""
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def vqa(self, image_path: str, question: str) -> dict:
        """
        Run Visual Question Answering.

        Args:
            image_path: Path to satellite image file
            question: Natural language question

        Returns:
            ToolResult dict with answer, confidence, etc.
        """
        with open(image_path, "rb") as f:
            files = {"image": (Path(image_path).name, f, "image/png")}
            data = {"question": question}

            resp = requests.post(
                f"{self.base_url}/vqa",
                files=files,
                data=data,
                timeout=self.timeout,
            )

        resp.raise_for_status()
        return resp.json()

    def caption(self, image_path: str, instruction: str | None = None) -> dict:
        """
        Generate scene description for a satellite image.

        Args:
            image_path: Path to satellite image file
            instruction: Optional custom captioning instruction

        Returns:
            ToolResult dict with caption, confidence, etc.
        """
        with open(image_path, "rb") as f:
            files = {"image": (Path(image_path).name, f, "image/png")}
            data = {}
            if instruction:
                data["instruction"] = instruction

            resp = requests.post(
                f"{self.base_url}/caption",
                files=files,
                data=data,
                timeout=self.timeout,
            )

        resp.raise_for_status()
        return resp.json()

    def grounding(self, image_path: str, query: str) -> dict:
        """
        Locate objects/features in a satellite image.

        Args:
            image_path: Path to satellite image file
            query: What to locate (e.g., "water body", "buildings")

        Returns:
            ToolResult dict with spatial_evidence, confidence, etc.
        """
        with open(image_path, "rb") as f:
            files = {"image": (Path(image_path).name, f, "image/png")}
            data = {"query": query}

            resp = requests.post(
                f"{self.base_url}/grounding",
                files=files,
                data=data,
                timeout=self.timeout,
            )

        resp.raise_for_status()
        return resp.json()


# ============================================================
# Convenience functions (drop-in replacements for direct imports)
# ============================================================
# Person 1 can use these exactly like the direct tool functions:
#   run_vqa(image_path, question) → dict
#   run_caption(image_path) → dict
#   run_grounding(image_path, query) → dict

_client: SatQueryClient | None = None


def _get_client() -> SatQueryClient:
    global _client
    if _client is None:
        import os
        server_url = os.getenv("SATQUERY_SERVER_URL", "http://localhost:8100")
        _client = SatQueryClient(server_url)
    return _client


def run_vqa(image_path: str, question: str) -> dict:
    """Remote VQA — same interface as backend.tools.vqa.run_vqa"""
    return _get_client().vqa(image_path, question)


def run_caption(image_path: str, instruction: str | None = None) -> dict:
    """Remote Caption — same interface as backend.tools.caption.run_caption"""
    return _get_client().caption(image_path, instruction)


def run_grounding(image_path: str, query: str) -> dict:
    """Remote Grounding — same interface as backend.tools.grounding.run_grounding"""
    return _get_client().grounding(image_path, query)


# CLI test
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Test SatQuery-RS client")
    parser.add_argument("--server", default="http://localhost:8100")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", default=None)
    parser.add_argument("--query", default=None)
    parser.add_argument("--task", choices=["vqa", "caption", "grounding"], default="vqa")
    args = parser.parse_args()

    client = SatQueryClient(args.server)

    # Health check
    print("Health:", json.dumps(client.health(), indent=2))

    if args.task == "vqa":
        result = client.vqa(args.image, args.question or "What is in this image?")
    elif args.task == "caption":
        result = client.caption(args.image)
    elif args.task == "grounding":
        result = client.grounding(args.image, args.query or "buildings")

    print(json.dumps(result, indent=2))
