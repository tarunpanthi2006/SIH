# SatQuery — Remote Sensing Intelligence System (SIH 2026 - Problem 167)

SatQuery is a multimodal, agentic intelligence system designed for remote sensing imagery. This repository contains the **Orchestration Backend (Person 1's architecture)** which routes queries, and the **Machine Learning Specialist Models (Person 2 & 3)** which perform inference.

## Architecture Highlights

- **Agentic Task Router**: Uses an LLM (`gemini-2.5-flash`) to semantically classify user queries into distinct `TaskType`s (e.g., Change Detection, VQA, Grounding, Multispectral).
- **Tool Registry**: A central hub where specialist models register their capabilities (required modalities, input bounds).
- **Execution Engine**: Generates a dynamic `ExecutionPlan` and manages multi-step tool chaining (e.g., ChangeMask -> VQA).
- **Input Validation**: Automatically extracts metadata from GeoTIFFs/PNGs and enforces spatial (CRS, overlap) and temporal constraints before execution.
- **Evidence Fusion**: Combines spatial outputs (masks, bounding boxes) and statistics into a standardized `EvidenceBundle`.

## Person 2: VLM Component (SatQuery-RS)

### Model
- **Base**: [LLaVA-1.5-7B](https://huggingface.co/llava-hf/llava-1.5-7b-hf) (LLaVA-1.5 architecture)
- **Adaptation**: QLoRA fine-tuned on BigEarthNet.txt (~40K samples, 3 epochs)
- **Result**: SatQuery-RS — remote-sensing adapted VLM
- **Quantization**: 4-bit NF4 with double quantization

### Capabilities
| Tool | Function | Description |
|---|---|---|
| VQA | `vqa` | Answer questions about satellite images |
| Caption | `caption` | Generate scene descriptions |
| Grounding | `grounding` | Locate objects with bounding boxes |
| Change-VQA | `change_vqa` | Interpret temporal changes |

## Getting Started

### Prerequisites
- Python 3.10+

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
2. The `SpecialistTool` classes in `backend/tools/` will automatically hook into the real model inference layers.

## Project Structure
- `backend/api/` - FastAPI routes and Pydantic schemas.
- `backend/agent/` - LLM Routing, Planning, and Executor.
- `backend/tools/` - Specialist tool interfaces and contracts.
- `backend/validation/` - Spatial, temporal, and modality validators.
- `models/` - P2 and P3 ML inference logic.
- `tests/` - Comprehensive pytest suite.
