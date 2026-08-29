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
   │  └── Grounding (run_ground) │
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
- **Base**: [GeoChat-7B](https://huggingface.co/MBZUAI/geochat-7B) (LLaVA-1.5 architecture)
- **Adaptation**: LoRA fine-tuned on BigEarthNet.txt (60K–100K samples)
- **Result**: SatQuery-RS — remote-sensing adapted VLM

### Capabilities
| Tool | Function | Description |
|---|---|---|
| VQA | `run_vqa(image, question)` | Answer questions about satellite images |
| Caption | `run_caption(image)` | Generate scene descriptions |
| Grounding | `run_grounding(image, query)` | Locate objects with bounding boxes |

### Quick Start
See [docs/development.md](docs/development.md) for full setup instructions.

```bash
pip install -r requirements.txt
python scripts/download_models.py --base
python -m backend.tools.vqa --test
```

## Team

| Person | Responsibility | Branch |
|---|---|---|
| Person 1 | Agent, API, Validation, Evidence Fusion | `main` |
| Person 2 | VLM, BigEarthNet, VQA, Caption, Grounding | `feature/vlm` |
| Person 3 | ChangeFormer, Optical/SAR, SkySense++, Prithvi | `feature/change` |
