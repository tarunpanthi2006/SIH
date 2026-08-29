"""
Tests for BigEarthNet.txt data pipeline.
Tests conversion and validation without requiring actual dataset download.
"""

import json
import os
import sys
import tempfile
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConversion:
    """Test the parquet → LLaVA instruction format conversion."""

    def test_binary_vqa_conversion(self):
        """Test binary VQA row conversion."""
        import pandas as pd
        from training.bigearthnet.convert import convert_binary_vqa

        row = pd.Series({
            "input": "Is there a water body present?",
            "output": "Yes",
            "type": "binary",
            "patch_id": "S2A_test_patch",
        }, name=0)

        result = convert_binary_vqa(row, "test/image.png")

        assert result is not None
        assert result["id"] == "ben_binary_0"
        assert result["image"] == "test/image.png"
        assert len(result["conversations"]) == 2
        assert result["conversations"][0]["from"] == "human"
        assert result["conversations"][1]["from"] == "gpt"
        assert "<image>" in result["conversations"][0]["value"]
        assert result["conversations"][1]["value"] == "Yes"

    def test_captioning_conversion(self):
        """Test captioning row conversion."""
        import pandas as pd
        from training.bigearthnet.convert import convert_captioning

        row = pd.Series({
            "input": "Describe this image.",
            "output": "Agricultural fields with mixed vegetation.",
            "type": "captioning",
            "patch_id": "S2A_test_patch",
        }, name=1)

        result = convert_captioning(row, "test/image.png")

        assert result is not None
        assert "cap" in result["id"]
        assert result["conversations"][1]["value"] == "Agricultural fields with mixed vegetation."

    def test_empty_output_skipped(self):
        """Test that rows with empty output are skipped."""
        import pandas as pd
        from training.bigearthnet.convert import convert_binary_vqa

        row = pd.Series({
            "input": "Question?",
            "output": "",
            "type": "binary",
            "patch_id": "S2A_test",
        }, name=0)

        result = convert_binary_vqa(row, "test/image.png")
        assert result is None


class TestValidation:
    """Test the instruction JSON validation."""

    def test_valid_sample_passes(self):
        from training.bigearthnet.validate import validate_schema

        sample = {
            "id": "test_001",
            "image": "test/image.png",
            "conversations": [
                {"from": "human", "value": "<image>\nWhat is this?"},
                {"from": "gpt", "value": "Agricultural land."},
            ],
        }

        errors = validate_schema(sample, 0)
        assert len(errors) == 0

    def test_missing_image_field(self):
        from training.bigearthnet.validate import validate_schema

        sample = {
            "id": "test_001",
            "conversations": [
                {"from": "human", "value": "<image>\nWhat is this?"},
                {"from": "gpt", "value": "Test."},
            ],
        }

        errors = validate_schema(sample, 0)
        assert any("image" in e for e in errors)

    def test_missing_image_token(self):
        from training.bigearthnet.validate import validate_schema

        sample = {
            "id": "test_001",
            "image": "test/image.png",
            "conversations": [
                {"from": "human", "value": "What is this?"},  # Missing <image>
                {"from": "gpt", "value": "Test."},
            ],
        }

        errors = validate_schema(sample, 0)
        assert any("<image>" in e for e in errors)


class TestSplits:
    """Test split generation."""

    def test_split_creation(self):
        from training.bigearthnet.splits import create_splits

        # Create temp instruction JSON
        samples = []
        for i in range(100):
            split = "train" if i < 70 else ("validation" if i < 85 else "test")
            samples.append({
                "id": f"test_{i}",
                "image": f"test/image_{i}.png",
                "conversations": [
                    {"from": "human", "value": f"<image>\nQuestion {i}?"},
                    {"from": "gpt", "value": f"Answer {i}."},
                ],
                "metadata": {
                    "task_type": "binary",
                    "split": split,
                },
            })

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.json"
            output_dir = Path(tmpdir) / "splits"

            with open(data_path, "w") as f:
                json.dump(samples, f)

            stats = create_splits(data_path, output_dir, debug_size=10, small_size=20)

            assert stats["train"] == 70
            assert stats["validation"] == 15
            assert stats["test"] == 15
            assert stats["debug"] == 10
            assert (output_dir / "train.json").exists()
            assert (output_dir / "validation.json").exists()
            assert (output_dir / "debug.json").exists()
