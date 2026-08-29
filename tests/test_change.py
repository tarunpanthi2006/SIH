import pytest
from unittest.mock import patch, MagicMock
from backend.tools.change import ChangeTool
from backend.tools.contracts import TaskType

@pytest.mark.asyncio
async def test_change_tool_missing_inputs():
    tool = ChangeTool()
    res = await tool.run(image_a="missing.png")
    assert res.is_error
    assert res.task == TaskType.CHANGE_DETECTION
    assert "image_b" in res.error_message or "required" in res.error_message

@pytest.mark.asyncio
async def test_change_tool_file_not_found():
    tool = ChangeTool()
    res = await tool.run(image_a="missing1.png", image_b="missing2.png")
    assert res.is_error
    assert "not found" in res.error_message
