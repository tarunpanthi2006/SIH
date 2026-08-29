# SatQuery — Agentic Remote-Sensing Intelligence System

> **Smart India Hackathon (SIH) 2026 — Problem Statement 167**

SatQuery is an agentic system that lets users query satellite imagery using natural language. It automatically routes queries to specialized remote-sensing AI models and returns evidence-backed answers.

## Architecture

```
User Query + Satellite Image
        ↓
   Person 1: Agent + API
   ├── Validation
   ├── Query Classification
   ├── Tool Registry
   ├── Evidence Fusion
   └── Confidence Scoring
        ↓
   ┌─────────────────────────────┐
   │     Specialist Tools        │
   ├─────────────────────────────┤
   │ Person 2: VLM               │
   │  ├── VQA (run_vqa)          │
   │  ├── Caption (run_caption)  │
   │  ├── Grounding (run_ground) │
   │  └── Change (run_change_vqa)│
   ├─────────────────────────────┤
   │ Person 3: Change + RS       │
   │  ├── ChangeFormer           │
   │  ├── Optical/SAR            │
   │  └── SkySense++/Prithvi     │
   └─────────────────────────────┘
        ↓
   Natural Language Answer + Evidence
```

## Person 2: VLM Component (SatQuery-RS)

### Model
- **Base**: [LLaVA-1.5-7B](https://huggingface.co/llava-hf/llava-1.5-7b-hf) (LLaVA-1.5 architecture)
- **Adaptation**: QLoRA fine-tuned on BigEarthNet.txt (~40K samples, 3 epochs)
- **Result**: SatQuery-RS — remote-sensing adapted VLM
- **Quantization**: 4-bit NF4 with double quantization

### Capabilities
| Tool | Function | Description |
|---|---|---|
| VQA | `run_vqa(image, question)` | Answer questions about satellite images |
| Caption | `run_caption(image)` | Generate scene descriptions |
| Grounding | `run_grounding(image, query)` | Locate objects with bounding boxes |
| Change-VQA | `run_change_vqa(img_a, img_b, question, mask)` | Interpret temporal changes |

### Quick Start
See [docs/development.md](docs/development.md) for full setup instructions.

```bash
pip install -r requirements.txt

# Schema tests (no GPU needed)
python -m backend.tools.vqa --test
python -m backend.tools.caption --test
python -m backend.tools.grounding --test
python -m backend.tools.change --test

# Start model server (requires GPU)
python -m backend.serve --host 0.0.0.0 --port 8100
```

### API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server health check |
| POST | `/vqa` | Visual Question Answering |
| POST | `/caption` | Image captioning |
| POST | `/grounding` | Object localization |
| POST | `/change-vqa` | Bi-temporal change interpretation |

See [docs/api.md](docs/api.md) for full API reference.

## Documentation
- [Architecture](docs/architecture.md) — System design and data flow
- [API Reference](docs/api.md) — Endpoint documentation with examples
- [Development Guide](docs/development.md) — Setup, training, and evaluation
- [Model Cards](docs/model_cards.md) — Model details and capabilities

## Team

| Person | Responsibility | Branch |
|---|---|---|
| Person 1 | Agent, API, Validation, Evidence Fusion | `main` |
| Person 2 | VLM, BigEarthNet, VQA, Caption, Grounding, Change-VQA | `feature/vlm` |
| Person 3 | ChangeFormer, Optical/SAR, SkySense++, Prithvi | `feature/change` |
