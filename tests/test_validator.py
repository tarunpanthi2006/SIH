"""
SatQuery — Validator Tests
"""

from pathlib import Path

from backend.api.schemas import Modality, TaskType
from backend.validation.validator import InputValidator
from backend.validation.modality import detect_modality, check_modality_compatibility
from backend.validation.spatial import check_spatial_compatibility, _compute_overlap_pct
from backend.validation.temporal import check_temporal_compatibility
from backend.api.schemas import ImageMetadata


class TestInputValidator:
    """Tests for the central InputValidator."""

    def test_valid_single_image(self, sample_png):
        validator = InputValidator()
        result = validator.validate([str(sample_png)])
        assert result.valid is True
        assert len(result.issues) == 0

    def test_file_not_found(self, nonexistent_path):
        validator = InputValidator()
        result = validator.validate([str(nonexistent_path)])
        assert result.valid is False
        assert any(i.code == "FILE_NOT_FOUND" for i in result.issues)

    def test_empty_file(self, empty_file):
        validator = InputValidator()
        result = validator.validate([str(empty_file)])
        assert result.valid is False
        assert any(i.code == "EMPTY_FILE" for i in result.issues)

    def test_unsupported_format(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("a,b,c")
        validator = InputValidator()
        result = validator.validate([str(p)])
        assert result.valid is False
        assert any(i.code == "UNSUPPORTED_FORMAT" for i in result.issues)

    def test_valid_pair(self, sample_png_pair):
        a, b = sample_png_pair
        validator = InputValidator()
        result = validator.validate([str(a), str(b)])
        assert result.valid is True

    def test_task_aware_modality_check(self, sample_png):
        """VQA with one image should pass modality checks."""
        validator = InputValidator()
        result = validator.validate([str(sample_png)], task=TaskType.VQA)
        assert result.valid is True

    def test_task_aware_insufficient_images(self, sample_png):
        """Change detection with 1 image should fail."""
        validator = InputValidator()
        result = validator.validate(
            [str(sample_png)], task=TaskType.CHANGE_DETECTION
        )
        assert result.valid is False
        assert any(i.code == "INSUFFICIENT_IMAGES" for i in result.issues)


class TestModalityDetection:
    """Tests for modality detection heuristics."""

    def test_optical_rgb(self):
        meta = ImageMetadata(path="x.png", bands=3, dtype="uint8")
        assert detect_modality(meta) == Modality.OPTICAL

    def test_sar_single_float(self):
        meta = ImageMetadata(path="x.tif", bands=1, dtype="float32")
        assert detect_modality(meta) == Modality.SAR

    def test_multispectral_many_bands(self):
        meta = ImageMetadata(path="x.tif", bands=13, dtype="uint16")
        assert detect_modality(meta) == Modality.MULTISPECTRAL

    def test_explicit_modality_preserved(self):
        meta = ImageMetadata(
            path="x.tif", bands=3, dtype="uint8", modality=Modality.SAR
        )
        assert detect_modality(meta) == Modality.SAR


class TestModalityCompatibility:
    """Tests for modality compatibility checks."""

    def test_vqa_accepts_optical(self):
        issues = check_modality_compatibility([Modality.OPTICAL], TaskType.VQA)
        assert len(issues) == 0

    def test_change_needs_two_images(self):
        issues = check_modality_compatibility(
            [Modality.OPTICAL], TaskType.CHANGE_DETECTION
        )
        assert any(i.code == "INSUFFICIENT_IMAGES" for i in issues)

    def test_optical_sar_needs_pair(self):
        issues = check_modality_compatibility(
            [Modality.OPTICAL, Modality.OPTICAL], TaskType.OPTICAL_SAR
        )
        assert any(i.code == "MISSING_MODALITY_PAIR" for i in issues)

    def test_optical_sar_valid_pair(self):
        issues = check_modality_compatibility(
            [Modality.OPTICAL, Modality.SAR], TaskType.OPTICAL_SAR
        )
        error_issues = [i for i in issues if i.severity == "error"]
        assert len(error_issues) == 0


class TestSpatialCompatibility:
    """Tests for spatial validation helpers."""

    def test_full_overlap(self):
        pct = _compute_overlap_pct(
            [0, 0, 10, 10], [0, 0, 10, 10]
        )
        assert pct == 100.0

    def test_no_overlap(self):
        pct = _compute_overlap_pct(
            [0, 0, 5, 5], [10, 10, 15, 15]
        )
        assert pct == 0.0

    def test_partial_overlap(self):
        pct = _compute_overlap_pct(
            [0, 0, 10, 10], [5, 5, 15, 15]
        )
        assert 20.0 < pct < 30.0  # 25% of the smaller box

    def test_no_crs_warning(self):
        meta_a = ImageMetadata(path="a.png")
        meta_b = ImageMetadata(path="b.png")
        issues = check_spatial_compatibility(meta_a, meta_b)
        assert any(i.code == "NO_CRS" for i in issues)


class TestTemporalCompatibility:
    """Tests for temporal validation."""

    def test_missing_dates_warning(self):
        meta_a = ImageMetadata(path="a.png")
        meta_b = ImageMetadata(path="b.png")
        issues = check_temporal_compatibility(meta_a, meta_b)
        assert any(i.code == "MISSING_DATES" for i in issues)

    def test_valid_temporal_order(self):
        meta_a = ImageMetadata(path="a.png", acquisition_date="2020-01-01")
        meta_b = ImageMetadata(path="b.png", acquisition_date="2023-01-01")
        issues = check_temporal_compatibility(meta_a, meta_b)
        # No ordering issues expected
        assert not any(i.code == "TEMPORAL_ORDER" for i in issues)

    def test_reversed_order_warning(self):
        meta_a = ImageMetadata(path="a.png", acquisition_date="2023-06-01")
        meta_b = ImageMetadata(path="b.png", acquisition_date="2020-01-01")
        issues = check_temporal_compatibility(meta_a, meta_b)
        assert any(i.code == "TEMPORAL_ORDER" for i in issues)
