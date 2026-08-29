"""
SatQuery — API Endpoint Tests

Tests the full pipeline via FastAPI TestClient.
"""

import os

import pytest
from fastapi.testclient import TestClient

# Ensure mock mode before importing app
os.environ["SATQUERY_MOCK_MODE"] = "true"

from backend.main import create_app
from backend.agent.registry import get_registry
from backend.tools.vqa import MockVQATool
from backend.tools.caption import MockCaptionTool
from backend.tools.grounding import MockGroundingTool
from backend.tools.change import ChangeTool
from backend.tools.optical_sar import OpticalSarTool
from backend.tools.multispectral import MultispectralTool


@pytest.fixture
def client():
    """
    Create a fresh TestClient.

    Tools are registered via the app lifespan, but the conftest
    `_reset_registry` fixture clears the global registry between tests.
    So we create the app fresh each time and use the context manager
    to trigger lifespan startup.
    """
    app = create_app()
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["mock_mode"] is True

    def test_health_has_version(self, client):
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert "version" in data


class TestToolsEndpoint:
    def test_list_tools(self, client):
        resp = client.get("/api/v1/tools")
        assert resp.status_code == 200
        tools = resp.json()
        assert isinstance(tools, list)
        assert len(tools) >= 6
        names = {t["name"] for t in tools}
        assert "vqa" in names
        assert "caption" in names
        assert "grounding" in names
        assert "change_detection" in names
        assert "optical_sar_analysis" in names
        assert "multispectral_analysis" in names


class TestTasksEndpoint:
    def test_list_tasks(self, client):
        resp = client.get("/api/v1/tasks")
        assert resp.status_code == 200
        tasks = resp.json()
        assert "vqa" in tasks
        assert "change_vqa" in tasks


class TestAnalyzeEndpoint:
    def test_vqa_single_image(self, client, sample_png):
        resp = client.post("/api/v1/analyze", json={
            "query": "What objects are visible in this image?",
            "images": [{"path": str(sample_png)}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["task"] == "vqa"
        assert len(data["answer"]) > 0
        assert 0.0 <= data["confidence"] <= 1.0
        assert "execution" in data
        assert "evidence" in data
        assert "request_id" in data

    def test_caption_single_image(self, client, sample_png):
        resp = client.post("/api/v1/analyze", json={
            "query": "Describe this scene.",
            "images": [{"path": str(sample_png)}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["task"] == "caption"
        assert len(data["answer"]) > 0

    def test_change_vqa_two_images(self, client, sample_png_pair):
        a, b = sample_png_pair
        resp = client.post("/api/v1/analyze", json={
            "query": "Has the built-up area increased?",
            "images": [
                {"path": str(a)},
                {"path": str(b)},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["task"] == "change_vqa"
        assert len(data["answer"]) > 0
        # Should have execution steps
        assert len(data["execution"]["steps"]) >= 1

    def test_file_not_found_returns_422(self, client):
        resp = client.post("/api/v1/analyze", json={
            "query": "What is here?",
            "images": [{"path": "/nonexistent/image.png"}],
        })
        assert resp.status_code == 422

    def test_missing_query_returns_422(self, client, sample_png):
        resp = client.post("/api/v1/analyze", json={
            "query": "",
            "images": [{"path": str(sample_png)}],
        })
        assert resp.status_code == 422

    def test_execution_trace_retrieval(self, client, sample_png):
        # Run an analysis first
        resp = client.post("/api/v1/analyze", json={
            "query": "What is here?",
            "images": [{"path": str(sample_png)}],
        })
        assert resp.status_code == 200
        request_id = resp.json()["request_id"]

        # Retrieve the trace
        resp2 = client.get(f"/api/v1/execution/{request_id}")
        assert resp2.status_code == 200
        trace = resp2.json()
        assert trace["request_id"] == request_id
        assert len(trace["steps"]) >= 1

    def test_execution_trace_not_found(self, client):
        resp = client.get("/api/v1/execution/does_not_exist")
        assert resp.status_code == 404


class TestAnalyzeResponseStructure:
    """Verify the full response schema matches frontend expectations."""

    def test_response_has_all_fields(self, client, sample_png):
        resp = client.post("/api/v1/analyze", json={
            "query": "Is there a river?",
            "images": [{"path": str(sample_png)}],
        })
        data = resp.json()
        required_keys = {
            "request_id", "task", "answer", "confidence",
            "evidence", "execution", "warnings",
        }
        assert required_keys.issubset(data.keys())

    def test_evidence_bundle_structure(self, client, sample_png):
        resp = client.post("/api/v1/analyze", json={
            "query": "Highlight the buildings.",
            "images": [{"path": str(sample_png)}],
        })
        evidence = resp.json()["evidence"]
        assert "primary_answer" in evidence
        assert "spatial_evidence" in evidence
        assert "provenance" in evidence

    def test_execution_summary_structure(self, client, sample_png):
        resp = client.post("/api/v1/analyze", json={
            "query": "Is this urban?",
            "images": [{"path": str(sample_png)}],
        })
        execution = resp.json()["execution"]
        assert "steps" in execution
        assert "models_used" in execution
        assert "total_duration_ms" in execution
