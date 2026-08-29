import pytest
from backend.tools.optical_sar import OpticalSarTool
from backend.tools.contracts import TaskType

@pytest.mark.asyncio
async def test_optical_sar_tool_missing_inputs():
    tool = OpticalSarTool()
    res = await tool.run(optical="missing.png")
    assert res.is_error
    assert res.task == TaskType.OPTICAL_SAR

@pytest.mark.asyncio
async def test_optical_sar_tool_file_not_found():
    tool = OpticalSarTool()
    res = await tool.run(optical="missing1.png", sar="missing2.png")
    assert res.is_error
    assert "not found" in res.error_message
