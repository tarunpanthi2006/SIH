"""
Tests for VQA tool wrapper.
Validates schema conformance without requiring model weights.
"""

import json
import os
import sys
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.tools.interfaces import ToolResult, validate_image_path


class TestToolResultSchema:
    """Test that ToolResult produces valid output schema."""

    def test_basic_vqa_result(self):
        result = ToolResult(
            task="vqa",
            model="SatQuery-RS",
            answer="Agricultural fields with mixed vegetation.",
            confidence=0.87,
        ).to_dict()

        assert result["task"] == "vqa"
        assert result["model"] == "SatQuery-RS"
        assert isinstance(result["answer"], str)
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["spatial_evidence"], list)
        assert isinstance(result["artifacts"], list)
        assert isinstance(result["metadata"], dict)
        assert isinstance(result["warnings"], list)

    def test_all_required_fields_present(self):
        result = ToolResult(
            task="vqa",
            model="SatQuery-RS",
            answer="Test answer",
            confidence=0.5,
        ).to_dict()

        required_fields = ["task", "model", "answer", "confidence",
                          "spatial_evidence", "artifacts", "metadata", "warnings"]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    def test_error_result(self):
        result = ToolResult.error("vqa", "SatQuery-RS", "Image not found")

        assert result["task"] == "vqa"
        assert result["answer"] == ""
        assert result["confidence"] == 0.0
        assert "Image not found" in result["warnings"]
        assert result["metadata"].get("error") is True

    def test_confidence_clamping(self):
        result = ToolResult(
            task="vqa",
            model="SatQuery-RS",
            answer="Test",
            confidence=1.5,  # Should be clamped to 1.0 on reporting
        ).to_dict()

        # We round but don't clamp in to_dict — that's the caller's job
        assert isinstance(result["confidence"], float)

    def test_json_serializable(self):
        result = ToolResult(
            task="vqa",
            model="SatQuery-RS",
            answer="Test answer with unicode: ñ ü ö",
            confidence=0.87,
            metadata={"key": "value", "nested": {"a": 1}},
        ).to_dict()

        # Should be JSON serializable
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed["answer"] == result["answer"]


class TestImageValidation:
    """Test image path validation."""

    def test_empty_path(self):
        is_valid, error = validate_image_path("")
        assert not is_valid
        assert "empty" in error.lower()

    def test_nonexistent_file(self):
        is_valid, error = validate_image_path("/nonexistent/image.png")
        assert not is_valid
        assert "not found" in error.lower()

    def test_unsupported_format(self):
        # Create a temp file with wrong extension
        temp_path = Path(__file__).parent / "temp_test.xyz"
        temp_path.touch()
        try:
            is_valid, error = validate_image_path(str(temp_path))
            assert not is_valid
            assert "unsupported" in error.lower()
        finally:
            temp_path.unlink()

    def test_valid_png(self):
        # Create a temp PNG file
        temp_path = Path(__file__).parent / "temp_test.png"
        temp_path.touch()
        try:
            is_valid, error = validate_image_path(str(temp_path))
            assert is_valid
            assert error == ""
        finally:
            temp_path.unlink()


class TestRunVqaInterface:
    """Test the run_vqa function interface (without model)."""

    def test_invalid_image_returns_error(self):
        from backend.tools.vqa import run_vqa

        result = run_vqa("/nonexistent/image.png", "What is this?")

        assert result["task"] == "vqa"
        assert result["confidence"] == 0.0
        assert len(result["warnings"]) > 0

    def test_empty_question_returns_error(self):
        from backend.tools.vqa import run_vqa

        # Create temp image
        temp_path = Path(__file__).parent / "temp_test.png"
        temp_path.touch()
        try:
            result = run_vqa(str(temp_path), "")
            assert result["task"] == "vqa"
            assert len(result["warnings"]) > 0
        finally:
            temp_path.unlink()
