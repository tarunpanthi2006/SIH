"""
SatQuery — Evidence Fusion

Merges outputs from one or more tools into a single ``EvidenceBundle``
with a primary answer, combined spatial evidence, statistics, and
per-tool provenance.
"""

from __future__ import annotations

from typing import Any

from backend.api.schemas import EvidenceBundle, SpatialEvidence, ToolResult


class EvidenceFusion:
    """
    Normalizes and merges tool results into one response bundle.

    Rules:
    - The primary answer comes from the *last* tool in the pipeline
      (which typically produces the final user-facing answer).
    - If only one tool ran, its answer is the primary answer.
    - Spatial evidence is concatenated (not invented).
    - Statistics are merged (non-overlapping keys); conflicts use the
      latest tool's value.
    - Provenance records which model produced which answer.
    """

    def fuse(
        self,
        tool_results: list[ToolResult],
        extra_metadata: dict[str, Any] | None = None,
    ) -> EvidenceBundle:
        """
        Fuse a list of tool results.

        Parameters
        ----------
        tool_results : list[ToolResult]
        extra_metadata : dict, optional

        Returns
        -------
        EvidenceBundle
        """
        if not tool_results:
            return EvidenceBundle()

        # -- Primary answer: last tool with a non-empty answer --
        primary_answer = ""
        for result in reversed(tool_results):
            if result.answer:
                primary_answer = result.answer
                break

        # -- Spatial evidence: collect from all tools --
        all_spatial: list[SpatialEvidence] = []
        for result in tool_results:
            all_spatial.extend(result.spatial_evidence)

        # -- Statistics: merge --
        merged_stats: dict[str, Any] = {}
        for result in tool_results:
            merged_stats.update(result.statistics)

        # -- Artifacts: collect unique --
        all_artifacts: list[str] = []
        seen_artifacts: set[str] = set()
        for result in tool_results:
            for art in result.artifacts:
                if art not in seen_artifacts:
                    all_artifacts.append(art)
                    seen_artifacts.add(art)

        # -- Provenance --
        provenance: list[dict[str, Any]] = []
        for result in tool_results:
            provenance.append({
                "task": result.task.value,
                "model": result.model,
                "answer": result.answer,
                "confidence": result.confidence,
                "has_spatial": len(result.spatial_evidence) > 0,
            })

        return EvidenceBundle(
            primary_answer=primary_answer,
            tool_results=tool_results,
            spatial_evidence=all_spatial,
            statistics=merged_stats,
            artifacts=all_artifacts,
            provenance=provenance,
        )
