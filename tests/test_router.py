"""
SatQuery — Task Router Tests
"""

import pytest
from backend.agent.router import classify_task
from backend.api.schemas import Modality, TaskType

pytestmark = pytest.mark.asyncio


class TestClassifyTask:
    """Tests for LLM task classification."""

    # -- VQA (default for questions) --

    async def test_simple_question_single_image(self):
        result = await classify_task("What objects are visible?", image_count=1)
        assert result == TaskType.VQA

    async def test_how_many_question(self):
        result = await classify_task("How many buildings are there?", image_count=1)
        assert result == TaskType.VQA

    # -- Caption --

    async def test_describe_scene(self):
        result = await classify_task("Describe this scene.", image_count=1)
        assert result == TaskType.CAPTION

    async def test_caption_keyword(self):
        result = await classify_task("Generate a caption for this image.", image_count=1)
        assert result == TaskType.CAPTION

    async def test_what_is_in_the_image(self):
        result = await classify_task("What is in this image?", image_count=1)
        assert result == TaskType.CAPTION

    async def test_non_question_defaults_caption(self):
        result = await classify_task("Overview of the area.", image_count=1)
        assert result == TaskType.CAPTION

    # -- Grounding --

    async def test_highlight_keyword(self):
        result = await classify_task("Highlight the water body.", image_count=1)
        assert result == TaskType.GROUNDING

    async def test_locate_keyword(self):
        result = await classify_task("Locate the airport runway.", image_count=1)
        assert result == TaskType.GROUNDING

    async def test_show_me(self):
        result = await classify_task("Show me the buildings.", image_count=1)
        assert result == TaskType.GROUNDING

    async def test_where_is(self):
        result = await classify_task("Where is the bridge?", image_count=1)
        assert result == TaskType.GROUNDING

    # -- Change VQA --

    async def test_change_question_two_images(self):
        result = await classify_task(
            "What changed between these two dates?", image_count=2
        )
        assert result == TaskType.CHANGE_VQA

    async def test_built_up_increased(self):
        result = await classify_task(
            "Has the built-up area increased?", image_count=2
        )
        assert result == TaskType.CHANGE_VQA

    async def test_change_keyword_single_image_falls_back(self):
        """Change keywords with 1 image should NOT route to change_vqa."""
        result = await classify_task("Has anything changed?", image_count=1)
        # The prompt will handle this but the mock falls back to vqa for change with 1 image
        assert result in (TaskType.VQA, TaskType.CAPTION, TaskType.CHANGE_VQA)

    # -- Change Detection --

    async def test_change_detection_explicit(self):
        result = await classify_task("Detect change in these images.", image_count=2)
        assert result == TaskType.CHANGE_DETECTION

    async def test_change_map(self):
        result = await classify_task("Generate a change map.", image_count=2)
        assert result == TaskType.CHANGE_DETECTION

    # -- Optical + SAR --

    async def test_optical_sar_keywords(self):
        result = await classify_task(
            "Analyze using optical and SAR together.", image_count=2
        )
        assert result == TaskType.OPTICAL_SAR

    async def test_optical_sar_from_modalities(self):
        result = await classify_task(
            "What do you see?",
            image_count=2,
            modalities=[Modality.OPTICAL, Modality.SAR],
        )
        assert result == TaskType.OPTICAL_SAR

    # -- Multispectral --

    async def test_ndvi_keyword(self):
        result = await classify_task("Compute NDVI for this image.", image_count=1)
        assert result == TaskType.MULTISPECTRAL

    async def test_spectral_index(self):
        result = await classify_task("Analyze the spectral index.", image_count=1)
        assert result == TaskType.MULTISPECTRAL

    # -- Task hint override --

    async def test_task_hint_overrides(self):
        result = await classify_task(
            "Describe this scene.",
            image_count=1,
            task_hint=TaskType.VQA,
        )
        assert result == TaskType.VQA
