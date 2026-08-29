"""
Tests for Grounding tool wrapper.
Validates bbox parsing and spatial evidence schema.
"""

import json
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.tools.interfaces import ToolResult
from models.grounding.inference import parse_bounding_boxes, format_spatial_evidence


class TestBboxParsing:
    """Test bounding box extraction from model output text."""

    def test_standard_bracket_format(self):
        text = "The water body is located at [0.32, 0.45, 0.78, 0.82]"
        bboxes = parse_bounding_boxes(text)
        assert len(bboxes) == 1
        assert bboxes[0] == [0.32, 0.45, 0.78, 0.82]

    def test_multiple_bboxes(self):
        text = "Buildings at [0.1, 0.2, 0.3, 0.4] and [0.5, 0.6, 0.7, 0.8]"
        bboxes = parse_bounding_boxes(text)
        assert len(bboxes) == 2

    def test_no_bboxes(self):
        text = "This is a description without any bounding boxes."
        bboxes = parse_bounding_boxes(text)
        assert len(bboxes) == 0

    def test_percentage_normalization(self):
        text = "Located at [32, 45, 78, 82]"
        bboxes = parse_bounding_boxes(text)
        assert len(bboxes) == 1
        assert all(0 <= c <= 1 for c in bboxes[0])

    def test_curly_brace_format(self):
        text = "Object at {0.2, 0.3, 0.5, 0.6}"
        bboxes = parse_bounding_boxes(text)
        assert len(bboxes) == 1

    def test_parenthesis_format(self):
        text = "Object at (0.2, 0.3, 0.5, 0.6)"
        bboxes = parse_bounding_boxes(text)
        assert len(bboxes) == 1

    def test_invalid_coords_filtered(self):
        # x1 > x2 should be swapped
        text = "Object at [0.8, 0.3, 0.2, 0.6]"
        bboxes = parse_bounding_boxes(text)
        if bboxes:
            assert bboxes[0][0] < bboxes[0][2]  # x1 < x2 after correction


class TestSpatialEvidence:
    """Test spatial evidence formatting."""

    def test_format_single_bbox(self):
        bboxes = [[0.1, 0.2, 0.5, 0.6]]
        evidence = format_spatial_evidence(bboxes, label="building")

        assert len(evidence) == 1
        assert evidence[0]["type"] == "bbox"
        assert len(evidence[0]["coordinates"]) == 4
        assert evidence[0]["label"] == "building"

    def test_format_empty_bboxes(self):
        evidence = format_spatial_evidence([], label="test")
        assert len(evidence) == 0


class TestGroundingToolInterface:
    """Test the run_grounding function interface (without model)."""

    def test_invalid_image_returns_error(self):
        from backend.tools.grounding import run_grounding

        result = run_grounding("/nonexistent/image.png", "water body")
        assert result["task"] == "grounding"
        assert result["confidence"] == 0.0
        assert len(result["warnings"]) > 0

    def test_empty_query_returns_error(self):
        from backend.tools.grounding import run_grounding

        temp_path = Path(__file__).parent / "temp_test.png"
        temp_path.touch()
        try:
            result = run_grounding(str(temp_path), "")
            assert result["task"] == "grounding"
            assert len(result["warnings"]) > 0
        finally:
            temp_path.unlink()

    def test_grounding_result_schema(self):
        result = ToolResult(
            task="grounding",
            model="SatQuery-RS",
            answer="Water body found in southeast region.",
            confidence=0.89,
            spatial_evidence=[
                {"type": "bbox", "coordinates": [0.6, 0.7, 0.9, 0.95], "label": "water body"}
            ],
        ).to_dict()

        assert result["task"] == "grounding"
        assert len(result["spatial_evidence"]) == 1
        assert result["spatial_evidence"][0]["type"] == "bbox"
        assert len(result["spatial_evidence"][0]["coordinates"]) == 4
