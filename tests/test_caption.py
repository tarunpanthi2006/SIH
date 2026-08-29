"""
Tests for Caption tool wrapper.
"""

import json
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.tools.interfaces import ToolResult


class TestCaptionSchema:
    """Test caption tool output schema."""

    def test_caption_result_schema(self):
        result = ToolResult(
            task="caption",
            model="SatQuery-RS",
            answer="An aerial view of agricultural fields with scattered settlements.",
            confidence=0.85,
        ).to_dict()

        assert result["task"] == "caption"
        assert result["model"] == "SatQuery-RS"
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["spatial_evidence"], list)

    def test_caption_json_serializable(self):
        result = ToolResult(
            task="caption",
            model="SatQuery-RS",
            answer="Coastal region with mixed urban and natural areas.",
            confidence=0.82,
            metadata={"inference_time_s": 12.5},
        ).to_dict()

        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed["task"] == "caption"


class TestRunCaptionInterface:
    """Test the run_caption function interface (without model)."""

    def test_invalid_image_returns_error(self):
        from backend.tools.caption import run_caption

        result = run_caption("/nonexistent/image.png")
        assert result["task"] == "caption"
        assert result["confidence"] == 0.0
        assert len(result["warnings"]) > 0
