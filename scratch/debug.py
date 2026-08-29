import asyncio
from backend.agent.registry import get_registry
from backend.agent.planner import WorkflowPlanner
from backend.agent.executor import PipelineExecutor
from backend.api.schemas import TaskType, ImageMetadata, Modality
import logging

logging.basicConfig(level=logging.DEBUG)

async def test_change():
    reg = get_registry()
    from backend.tools.change import ChangeTool
    reg.register(ChangeTool())
    
    planner = WorkflowPlanner(reg)
    meta1 = ImageMetadata(path="/tmp/a.png", modality=Modality.OPTICAL)
    meta2 = ImageMetadata(path="/tmp/b.png", modality=Modality.OPTICAL)
    
    plan = await planner.plan(
        task=TaskType.CHANGE_DETECTION,
        image_paths=["/tmp/a.png", "/tmp/b.png"],
        query="what changed?",
        metadata=[meta1, meta2],
        chat_history=None
    )
    print("Plan:", plan)
    
    executor = PipelineExecutor(reg)
    res, trace = await executor.execute(plan)
    print(res)

if __name__ == "__main__":
    asyncio.run(test_change())
