import pytest
from backend.tools.multispectral import MultispectralTool
from backend.tools.contracts import TaskType

@pytest.mark.asyncio
async def test_multispectral_tool_missing_inputs():
    tool = MultispectralTool()
    res = await tool.run()
    assert res.is_error
    assert res.task == TaskType.MULTISPECTRAL

@pytest.mark.asyncio
async def test_multispectral_tool_file_not_found():
    tool = MultispectralTool()
    res = await tool.run(image="missing.tif")
    assert res.is_error
    assert "not found" in res.error_message
