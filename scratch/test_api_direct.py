import asyncio
import os
from backend.agent.registry import get_registry
from backend.api.schemas import AnalyzeRequest, ImageInput
from backend.api.routes import analyze

async def run_local():
    # Setup test file
    import io
    from PIL import Image
    os.makedirs("uploads", exist_ok=True)
    img = Image.new('RGB', (100, 100), color=(255, 0, 0))
    filepath = os.path.abspath("uploads/test_vqa.png")
    img.save(filepath, format='PNG')
    
    req = AnalyzeRequest(
        query="How many buildings are in this residential area?",
        images=[ImageInput(path=filepath)]
    )
    
    try:
        res = await analyze(req)
        print(res.model_dump_json(indent=2))
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_local())
