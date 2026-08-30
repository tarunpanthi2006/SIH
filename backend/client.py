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
import io
import json
import logging
import os
from pathlib import Path
from typing import Any

import requests
from PIL import Image

logger = logging.getLogger(__name__)


def _optimize_image(image_path: str, max_size: int = 1024) -> tuple[str, io.BytesIO, str]:
    """
    Compresses and resizes an image in-memory before sending over the network.
    This prevents massive network latency if the Agent tries to send a 4K image.
    
    Returns:
        tuple of (filename, BytesIO stream, mime_type)
    """
    try:
        img = Image.open(image_path)
        
        # Resize if larger than max_size while maintaining aspect ratio
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
        # Convert to RGB to discard alpha channel bloat if present
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        # Save to memory buffer as high-quality JPEG
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        
        filename = Path(image_path).with_suffix(".jpg").name
        return filename, buffer, "image/jpeg"
        
    except Exception as e:
        logger.warning(f"Image optimization failed for {image_path}: {e}. Falling back to raw file.")
        # Fallback to reading raw file if PIL fails
        f = open(image_path, "rb")
        return Path(image_path).name, f, "image/png"


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
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"status": "unreachable", "error": str(e)}

    def vqa(self, image_path: str, question: str) -> dict:
        """
        Run Visual Question Answering.

        Args:
            image_path: Path to satellite image file
            question: Natural language question

        Returns:
            ToolResult dict with answer, confidence, etc.
        """
        try:
            filename, file_stream, mime = _optimize_image(image_path)
            
            try:
                files = {"image": (filename, file_stream, mime)}
                data = {"question": question}

                resp = requests.post(
                    f"{self.base_url}/vqa",
                    files=files,
                    data=data,
                    timeout=self.timeout,
                )
            finally:
                file_stream.close()

            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return _format_error("vqa", str(e))

    def caption(self, image_path: str, instruction: str | None = None) -> dict:
        """
        Generate scene description for a satellite image.

        Args:
            image_path: Path to satellite image file
            instruction: Optional custom captioning instruction

        Returns:
            ToolResult dict with caption, confidence, etc.
        """
        try:
            filename, file_stream, mime = _optimize_image(image_path)
            
            try:
                files = {"image": (filename, file_stream, mime)}
                data = {}
                if instruction:
                    data["instruction"] = instruction

                resp = requests.post(
                    f"{self.base_url}/caption",
                    files=files,
                    data=data,
                    timeout=self.timeout,
                )
            finally:
                file_stream.close()

            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return _format_error("caption", str(e))

    def grounding(self, image_path: str, query: str) -> dict:
        """
        Locate objects/features in a satellite image.

        Args:
            image_path: Path to satellite image file
            query: What to locate (e.g., "water body", "buildings")

        Returns:
            ToolResult dict with spatial_evidence, confidence, etc.
        """
        try:
            filename, file_stream, mime = _optimize_image(image_path)
            
            try:
                files = {"image": (filename, file_stream, mime)}
                data = {"query": query}

                resp = requests.post(
                    f"{self.base_url}/grounding",
                    files=files,
                    data=data,
                    timeout=self.timeout,
                )
            finally:
                file_stream.close()

            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return _format_error("grounding", str(e))

    def change_vqa(
        self,
        image_before: str,
        image_after: str,
        question: str = "What changes are visible between the two time periods?",
        change_mask: str | None = None,
    ) -> dict:
        """
        Bi-temporal change interpretation.

        Args:
            image_before: Path to T1 (before) image
            image_after: Path to T2 (after) image
            question: Question about the change
            change_mask: Optional path to P3's binary change mask

        Returns:
            ToolResult dict with change interpretation
        """
        try:
            name_a, stream_a, mime_a = _optimize_image(image_before)
            name_b, stream_b, mime_b = _optimize_image(image_after)
            
            try:
                files = {
                    "image_before": (name_a, stream_a, mime_a),
                    "image_after": (name_b, stream_b, mime_b)
                }

                if change_mask:
                    # Do NOT optimize the change_mask. It is a binary segmentation mask.
                    # JPEG compression or resizing would destroy the exact pixel boundaries.
                    fm = open(change_mask, "rb")
                    files["change_mask"] = (Path(change_mask).name, fm, "image/png")

                data = {"question": question}

                resp = requests.post(
                    f"{self.base_url}/change-vqa",
                    files=files,
                    data=data,
                    timeout=self.timeout,
                )

                if change_mask:
                    fm.close()
            finally:
                stream_a.close()
                stream_b.close()

            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return _format_error("change_vqa", str(e))


def _format_error(task: str, error_msg: str) -> dict:
    """Helper to return a standardized ToolResult error dict."""
    return {
        "task": task,
        "model": "SatQueryClient",
        "answer": "",
        "confidence": 0.0,
        "spatial_evidence": [],
        "artifacts": [],
        "metadata": {"error": True},
        "warnings": [f"API Connection Error: {error_msg}"]
    }


# ============================================================
# Convenience functions (drop-in replacements for direct imports)
# ============================================================
# Person 1 can use these exactly like the direct tool functions:
#   run_vqa(image_path, question) → dict
#   run_caption(image_path) → dict
#   run_grounding(image_path, query) → dict
#   run_change_vqa(image_a, image_b, question, mask) → dict

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


def run_change_vqa(
    image_a: str,
    image_b: str,
    question: str = "What changes are visible?",
    change_mask: str | None = None,
) -> dict:
    """Remote Change-VQA — same interface as backend.tools.change.run_change_vqa"""
    return _get_client().change_vqa(image_a, image_b, question, change_mask)


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
