"""
SatQuery-RS — Model Serving API
==================================
FastAPI server that wraps VQA, Caption, and Grounding tools
as HTTP endpoints. This is what runs on your cloud GPU or local machine.

Person 1's backend calls these endpoints via HTTP.

Usage:
    # Local (CPU offload mode on RTX 4050)
    SATQUERY_MODE=cpu-offload python -m backend.serve

    # Cloud GPU (4-bit quantized, fast)
    SATQUERY_MODE=quantized-4bit python -m backend.serve --host 0.0.0.0 --port 8100

    # Then call from anywhere:
    curl -X POST http://your-server:8100/vqa \
        -F "image=@satellite.png" \
        -F "question=What land cover is visible?"
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Lifespan: pre-load model at startup
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the VLM model once at server startup."""
    logger.info("🚀 Starting SatQuery-RS Model Server...")
    logger.info(f"   Mode: {os.getenv('SATQUERY_MODE', 'cpu-offload')}")

    try:
        from models.vqa.model import get_model
        vlm = get_model()
        logger.info(f"✅ Model loaded: {vlm.model_name}")
        logger.info(f"   Device: {vlm.device}")
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        logger.info("Server will start but inference calls will fail.")

    yield  # Server runs

    logger.info("Shutting down SatQuery-RS server...")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="SatQuery-RS Model Server",
    description="Remote Sensing VLM API — VQA, Captioning, and Grounding",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow CORS for Person 1's frontend/backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temp directory for uploaded images
UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="satquery_"))


async def save_upload(file: UploadFile) -> str:
    """Save an uploaded file to a temp location and return the path."""
    suffix = Path(file.filename or "image.png").suffix or ".png"
    temp_path = UPLOAD_DIR / f"{int(time.time() * 1000)}{suffix}"

    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return str(temp_path)


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
async def health():
    """Health check — verify model is loaded."""
    try:
        from models.vqa.model import SatQueryVLM
        vlm = SatQueryVLM.get_instance()
        return {
            "status": "healthy",
            "model": vlm.model_name,
            "loaded": vlm.is_loaded,
            "mode": vlm.config.get("mode", "unknown"),
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/")
async def root():
    return {
        "service": "SatQuery-RS Model Server",
        "endpoints": ["/vqa", "/caption", "/grounding", "/health"],
    }


# ============================================================
# VQA Endpoint
# ============================================================

@app.post("/vqa")
async def vqa_endpoint(
    image: UploadFile = File(..., description="Satellite/RS image"),
    question: str = Form(..., description="Question about the image"),
):
    """
    Visual Question Answering on a satellite image.

    Upload an image and ask a question — get a natural language answer.
    """
    image_path = await save_upload(image)

    try:
        from backend.tools.vqa import run_vqa
        result = run_vqa(image_path, question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up uploaded file
        Path(image_path).unlink(missing_ok=True)


# ============================================================
# Caption Endpoint
# ============================================================

@app.post("/caption")
async def caption_endpoint(
    image: UploadFile = File(..., description="Satellite/RS image"),
    instruction: str = Form(
        default="Describe this satellite image in detail.",
        description="Optional custom captioning instruction",
    ),
):
    """
    Generate a detailed scene description of a satellite image.
    """
    image_path = await save_upload(image)

    try:
        from backend.tools.caption import run_caption
        result = run_caption(image_path, instruction=instruction)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(image_path).unlink(missing_ok=True)


# ============================================================
# Grounding Endpoint
# ============================================================

@app.post("/grounding")
async def grounding_endpoint(
    image: UploadFile = File(..., description="Satellite/RS image"),
    query: str = Form(..., description="What to locate, e.g. 'water body'"),
):
    """
    Locate a specific object or feature in a satellite image.

    Returns bounding box coordinates [x1, y1, x2, y2] normalized to [0, 1].
    """
    image_path = await save_upload(image)

    try:
        from backend.tools.grounding import run_grounding
        result = run_grounding(image_path, query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(image_path).unlink(missing_ok=True)


# ============================================================
# Main
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="SatQuery-RS Model Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8100, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev)")
    args = parser.parse_args()

    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(
        "backend.serve:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
