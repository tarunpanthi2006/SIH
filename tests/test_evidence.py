"""
SatQuery — Evidence Fusion & Confidence Tests
"""

import pytest

from backend.api.schemas import (
    EvidenceBundle,
    ImageMetadata,
    Modality,
    SpatialEvidence,
    SpatialEvidenceType,
    TaskType,
    ToolResult,
    ValidationResult,
)
from backend.evidence.confidence import ConfidenceEngine
from backend.evidence.fusion import EvidenceFusion


class TestEvidenceFusion:
    """Tests for the evidence fusion layer."""

    def test_empty_results(self):
        fusion = EvidenceFusion()
        bundle = fusion.fuse([])
        assert bundle.primary_answer == ""
        assert len(bundle.tool_results) == 0

    def test_single_result(self):
        result = ToolResult(
            task=TaskType.VQA,
            model="test",
            answer="There are buildings.",
            confidence=0.9,
        )
        fusion = EvidenceFusion()
        bundle = fusion.fuse([result])
        assert bundle.primary_answer == "There are buildings."
        assert len(bundle.provenance) == 1

    def test_multiple_results_last_answer(self):
        r1 = ToolResult(
            task=TaskType.CHANGE_DETECTION,
            model="ChangeFormer",
            answer="Change detected.",
            confidence=0.85,
        )
        r2 = ToolResult(
            task=TaskType.CHANGE_VQA,
            model="SatQuery-RS",
            answer="Urban area has expanded.",
            confidence=0.92,
        )
        fusion = EvidenceFusion()
        bundle = fusion.fuse([r1, r2])
        assert bundle.primary_answer == "Urban area has expanded."
        assert len(bundle.provenance) == 2

    def test_spatial_evidence_merged(self):
        r1 = ToolResult(
            task=TaskType.GROUNDING,
            model="GeoChat",
            answer="Found region.",
            confidence=0.9,
            spatial_evidence=[
                SpatialEvidence(
                    type=SpatialEvidenceType.BBOX,
                    coordinates=[0.1, 0.2, 0.3, 0.4],
                ),
            ],
        )
        r2 = ToolResult(
            task=TaskType.CHANGE_DETECTION,
            model="ChangeFormer",
            answer="Change mask.",
            confidence=0.88,
            spatial_evidence=[
                SpatialEvidence(
                    type=SpatialEvidenceType.MASK,
                    path="/mask.png",
                ),
            ],
        )
        fusion = EvidenceFusion()
        bundle = fusion.fuse([r1, r2])
        assert len(bundle.spatial_evidence) == 2

    def test_artifacts_deduplicated(self):
        r1 = ToolResult(
            task=TaskType.VQA, model="a", answer="x",
            artifacts=["/a.png", "/b.png"],
        )
        r2 = ToolResult(
            task=TaskType.VQA, model="b", answer="y",
            artifacts=["/b.png", "/c.png"],
        )
        fusion = EvidenceFusion()
        bundle = fusion.fuse([r1, r2])
        assert bundle.artifacts == ["/a.png", "/b.png", "/c.png"]


class TestConfidenceEngine:
    """Tests for the confidence computation."""

    def test_empty_results_zero(self):
        engine = ConfidenceEngine()
        assert engine.compute([]) == 0.0

    def test_single_high_confidence(self):
        result = ToolResult(
            task=TaskType.VQA, model="test",
            answer="answer", confidence=0.95,
        )
        engine = ConfidenceEngine()
        score = engine.compute([result])
        assert 0.4 < score < 1.0  # model_confidence dominates

    def test_validation_warnings_reduce_score(self):
        result = ToolResult(
            task=TaskType.VQA, model="test",
            answer="answer", confidence=0.95,
        )
        from backend.api.schemas import ValidationIssue
        val_clean = ValidationResult(valid=True, warnings=[])
        val_warned = ValidationResult(
            valid=True,
            warnings=[
                ValidationIssue(code="W1", message="w1", severity="warning"),
                ValidationIssue(code="W2", message="w2", severity="warning"),
                ValidationIssue(code="W3", message="w3", severity="warning"),
            ],
        )
        engine = ConfidenceEngine()
        score_clean = engine.compute([result], validation=val_clean)
        score_warned = engine.compute([result], validation=val_warned)
        assert score_warned < score_clean

    def test_confidence_bounded(self):
        result = ToolResult(
            task=TaskType.VQA, model="test",
            answer="a", confidence=1.0,
            spatial_evidence=[
                SpatialEvidence(
                    type=SpatialEvidenceType.BBOX,
                    coordinates=[0, 0, 1, 1],
                )
            ],
        )
        engine = ConfidenceEngine()
        score = engine.compute([result])
        assert 0.0 <= score <= 1.0
