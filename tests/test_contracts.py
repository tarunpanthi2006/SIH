import pytest
from backend.tools.contracts import SpecialistOutput, make_success, make_error, TaskType, EvidenceType, SpatialEvidence, Artifact

def test_make_success():
    out = make_success(
        task=TaskType.CHANGE_DETECTION,
        model="TestModel",
        answer="Change detected."
    )
    assert not out.is_error
    assert out.task == TaskType.CHANGE_DETECTION
    assert out.model == "TestModel"
    assert out.answer == "Change detected."

def test_make_error():
    out = make_error(
        task=TaskType.OPTICAL_SAR,
        model="TestModel",
        error_message="File not found"
    )
    assert out.is_error
    assert out.error_message == "File not found"

def test_serialization():
    out = make_success(
        task=TaskType.MULTISPECTRAL,
        model="M1",
        answer="A",
        spatial_evidence=[SpatialEvidence(type=EvidenceType.MASK, path="path.png")]
    )
    data = out.model_dump()
    assert data["task"] == "multispectral"
    assert data["spatial_evidence"][0]["type"] == "mask"
    
    # deserialize
    out2 = SpecialistOutput.model_validate(data)
    assert out2.model == "M1"
