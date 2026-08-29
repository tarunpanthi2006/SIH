# SatQuery — Remote Sensing Intelligence System (SIH 2026 - Problem 167)

SatQuery is a multimodal, agentic intelligence system designed for remote sensing imagery. This repository contains the **Orchestration Backend (Person 1's architecture)**, which dynamically routes natural-language user queries to the appropriate ML specialist models, handles cross-modal spatial/temporal validation, executes multi-step plans, and fuses evidence into a structured response.

## Architecture Highlights

- **Agentic Task Router**: Uses an LLM (`gemini-2.5-flash`) to semantically classify user queries into distinct `TaskType`s (e.g., Change Detection, VQA, Grounding, Multispectral).
- **Tool Registry**: A central hub where specialist models (from Person 2 & 3) register their capabilities (required modalities, input bounds).
- **Execution Engine**: Generates a dynamic `ExecutionPlan` and manages multi-step tool chaining (e.g., ChangeMask -> VQA).
- **Input Validation**: Automatically extracts metadata from GeoTIFFs/PNGs and enforces spatial (CRS, overlap) and temporal constraints before execution.
- **Evidence Fusion**: Combines spatial outputs (masks, bounding boxes) and statistics into a standardized `EvidenceBundle`.
- **Structured API**: Robust FastAPI server exposing a clean `/api/v1/analyze` endpoint.

## Getting Started

### Prerequisites
- Python 3.10+
- (Optional) Redis (if caching is enabled in future)

### Installation
1. Clone the repository and navigate to the root directory.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment:
   ```bash
   cp .env.example .env
   # Add your GEMINI_API_KEY to the .env file
   ```

### Running the Server
Use the provided script to start the FastAPI server:
```bash
bash scripts/run_backend.sh
```
The server will start on `http://0.0.0.0:8000`. You can view the interactive Swagger API documentation at `http://localhost:8000/docs`.

### Testing
Run the comprehensive test suite (80+ integration and unit tests) via pytest:
```bash
source .venv/bin/activate
PYTHONPATH=$(pwd) pytest -v tests/
```

## Integration with P2/P3 Models

The backend currently runs in `MOCK_MODE` (or uses stub interfaces) for the specialist models. To integrate real models:
1. Ensure `SATQUERY_MOCK_MODE=false` in `.env`.
2. Ensure Person 2's VQA/Grounding code replaces the `MockVQATool` in `backend/tools/vqa.py`.
3. Ensure Person 3's checkpoints (`ChangeFormer`, `Prithvi-EO`, `SkySense++`) are placed in the `checkpoints/` directory.

## Project Structure
- `backend/api/` - FastAPI routes and Pydantic schemas.
- `backend/agent/` - LLM Routing, Planning, and Executor.
- `backend/tools/` - Specialist tool interfaces and contracts.
- `backend/validation/` - Spatial, temporal, and modality validators.
- `backend/evidence/` - Confidence scoring and fusion engine.
- `tests/` - Comprehensive pytest suite.
