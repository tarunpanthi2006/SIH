"""
SatQuery — Tool Registry Tests
"""

import pytest

from backend.agent.registry import ToolRegistry
from backend.api.schemas import TaskType
from backend.tools.vqa import MockVQATool
from backend.tools.caption import MockCaptionTool


class TestToolRegistry:
    """Tests for the tool registry."""

    def test_register_and_get(self):
        reg = ToolRegistry()
        tool = MockVQATool()
        reg.register(tool)
        assert reg.has("vqa")
        retrieved = reg.get("vqa")
        assert retrieved.name == "vqa"

    def test_get_missing_raises(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError, match="not_a_tool"):
            reg.get("not_a_tool")

    def test_list_tools(self, mock_registry):
        tools = mock_registry.list_tools()
        assert len(tools) == 7
        names = {t.name for t in tools}
        assert "vqa" in names
        assert "change_detection" in names
        assert "optical_sar" in names

    def test_list_for_task(self, mock_registry):
        vqa_tools = mock_registry.list_for_task(TaskType.VQA)
        assert len(vqa_tools) == 1
        assert vqa_tools[0].name == "vqa"

    def test_overwrite_tool(self):
        reg = ToolRegistry()
        reg.register(MockVQATool())
        reg.register(MockVQATool())  # should overwrite without error
        assert reg.has("vqa")

    def test_tool_names(self, mock_registry):
        names = mock_registry.tool_names
        assert isinstance(names, list)
        assert names == sorted(names)  # should be sorted
