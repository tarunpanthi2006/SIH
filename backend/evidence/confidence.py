"""
SatQuery — Confidence Engine

Computes a transparent confidence score from measurable signals.
No LLM-generated randomness — only observable factors.

The method is designed to be replaceable; swap this class to change
the confidence algorithm without touching the rest of the system.
"""

from __future__ import annotations

from backend.api.schemas import EvidenceBundle, ToolResult, ValidationResult


class ConfidenceEngine:
    """
    Weighted confidence computation.

    ============================  ======  ================================
    Signal                        Weight  Source
    ============================  ======  ================================
    Model confidence              0.40    Mean of per-tool confidence
    Input quality                 0.15    Validation warnings count
    Evidence availability         0.15    Spatial evidence present? Count?
    Cross-tool agreement          0.15    Answer similarity heuristic
    Metadata completeness         0.15    CRS / date / sensor present?
    ============================  ======  ================================
    """

    W_MODEL = 0.40
    W_INPUT = 0.15
    W_EVIDENCE = 0.15
    W_AGREEMENT = 0.15
    W_METADATA = 0.15

    def compute(
        self,
        tool_results: list[ToolResult],
        validation: ValidationResult | None = None,
        evidence: EvidenceBundle | None = None,
    ) -> float:
        """
        Compute overall confidence in [0.0, 1.0].

        Parameters
        ----------
        tool_results : list[ToolResult]
        validation : ValidationResult, optional
        evidence : EvidenceBundle, optional

        Returns
        -------
        float
        """
        if not tool_results:
            return 0.0

        s_model = self._model_confidence(tool_results)
        s_input = self._input_quality(validation)
        s_evidence = self._evidence_availability(tool_results, evidence)
        s_agreement = self._cross_tool_agreement(tool_results)
        s_metadata = self._metadata_completeness(validation)

        score = (
            self.W_MODEL * s_model
            + self.W_INPUT * s_input
            + self.W_EVIDENCE * s_evidence
            + self.W_AGREEMENT * s_agreement
            + self.W_METADATA * s_metadata
        )

        return round(min(max(score, 0.0), 1.0), 3)

    # ------------------------------------------------------------------ #
    # Signal functions (each returns 0–1)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _model_confidence(results: list[ToolResult]) -> float:
        """Mean of per-tool model confidence."""
        if not results:
            return 0.0
        total = sum(r.confidence for r in results)
        return total / len(results)

    @staticmethod
    def _input_quality(validation: ValidationResult | None) -> float:
        """Penalize based on validation warning count."""
        if validation is None:
            return 0.5  # no info → neutral
        warning_count = len(validation.warnings)
        if warning_count == 0:
            return 1.0
        # Each warning reduces quality; cap at 0.2
        return max(1.0 - warning_count * 0.15, 0.2)

    @staticmethod
    def _evidence_availability(
        results: list[ToolResult],
        evidence: EvidenceBundle | None,
    ) -> float:
        """Higher when spatial evidence and multiple sources exist."""
        spatial_count = 0
        for r in results:
            spatial_count += len(r.spatial_evidence)
        if evidence:
            spatial_count += len(evidence.spatial_evidence)

        if spatial_count == 0:
            return 0.3  # no spatial evidence → low but not zero
        if spatial_count == 1:
            return 0.7
        return min(0.7 + spatial_count * 0.05, 1.0)

    @staticmethod
    def _cross_tool_agreement(results: list[ToolResult]) -> float:
        """
        Basic agreement heuristic.

        With only one tool there's no disagreement to measure → 0.7.
        With multiple tools, check whether confidence values are close.
        """
        if len(results) <= 1:
            return 0.7  # single tool → neutral

        confidences = [r.confidence for r in results if r.confidence > 0]
        if not confidences:
            return 0.5
        spread = max(confidences) - min(confidences)
        # Low spread → high agreement
        return max(1.0 - spread, 0.3)

    @staticmethod
    def _metadata_completeness(
        validation: ValidationResult | None,
    ) -> float:
        """Higher when CRS, acquisition date, and sensor are present."""
        if validation is None or not validation.metadata:
            return 0.3

        total_fields = 0
        present_fields = 0
        for meta in validation.metadata:
            total_fields += 3  # CRS, date, sensor
            if meta.crs:
                present_fields += 1
            if meta.acquisition_date:
                present_fields += 1
            if meta.sensor:
                present_fields += 1

        if total_fields == 0:
            return 0.3
        return max(present_fields / total_fields, 0.3)
