# SatQuery-RS — System Architecture

## Overview

SatQuery is an agentic remote-sensing intelligence system built for SIH 2026 (Problem Statement 167). It lets users query satellite imagery using natural language and returns evidence-backed answers with spatial grounding.

## High-Level Architecture

```
User Query + Satellite Image
        ↓
   Person 1: Agent + API
   ├── Validation & Query Classification
   ├── Tool Registry & Routing
   ├── Evidence Fusion
   └── Confidence Scoring
        ↓
   ┌──────────────────────────────────┐
   │       Specialist Tools           │
   ├──────────────────────────────────┤
   │ Person 2: VLM (SatQuery-RS)     │
   │  ├── VQA      → run_vqa()       │
   │  ├── Caption  → run_caption()   │
   │  ├── Ground   → run_grounding() │
   │  └── Change   → run_change_vqa()│
   ├──────────────────────────────────┤
   │ Person 3: Change Detection + RS  │
   │  ├── ChangeFormer                │
   │  ├── Optical/SAR Fusion          │
   │  └── SkySense++ / Prithvi        │
   └──────────────────────────────────┘
        ↓
   Natural Language Answer + Evidence
```

## Person 2: VLM Component

### Model Architecture

```
Satellite Image (336×336 RGB)
        ↓
CLIP ViT-L/14 @ 336px (frozen)
        ↓
576 visual tokens (24×24 grid)
        ↓
Multi-Modal Projector (linear)
        ↓
   ┌─────────────────────────┐
   │ LLaVA-1.5 7B (Vicuna)   │
   │ + LoRA Adapters (2.05%)  │
   │   - q_proj, k_proj       │
   │   - v_proj, o_proj       │
   │   - rank=64, alpha=128   │
   └─────────────────────────┘
        ↓
Text Generation (answer, bbox, caption)
```

### Inference Modes

| Mode | VRAM | RAM | Speed | Use Case |
|---|---|---|---|---|
| `quantized-4bit` | ~5 GB | ~4 GB | Fast | RTX 4050 (tight) |
| `cpu-offload` | ~5 GB | ~20 GB | Medium | RTX 4050 (safe) |
| `full-fp16` | ~14 GB | ~4 GB | Fastest | Cloud GPU (A100/H100) |

### Training Pipeline

```
BigEarthNet Parquet (9.5M rows)
        ↓
Stratified Sampling (80K samples)
        ↓
LLaVA JSON Conversion (convert.py)
        ↓
Train/Val/Test Split (39K/500/500)
        ↓
QLoRA Fine-tuning (4-bit NF4)
        ↓
LoRA Adapter (~300 MB)
        ↓
SatQuery-RS Adapted Model
```

## Data Flow

### VQA / Caption / Grounding

```
P1 Agent                            P2 Model Server
   │                                      │
   │  POST /vqa {image, question}         │
   │─────────────────────────────────────>│
   │                                      │
   │      ToolResult JSON                 │
   │<─────────────────────────────────────│
   │                                      │
```

### Change-VQA (P2 + P3 Collaboration)

```
P1 Agent
   │
   ├── POST P3: ChangeFormer(T1, T2)
   │       ↓
   │   change_mask.png
   │       ↓
   ├── POST P2: /change-vqa {T1, T2, mask, question}
   │       ↓
   │   ToolResult JSON with semantic interpretation
   │
```

## Common Output Contract

Every tool returns a standardized `ToolResult`:

```json
{
    "task": "vqa | caption | grounding | change_vqa",
    "model": "SatQuery-RS",
    "answer": "natural language answer",
    "confidence": 0.0-1.0,
    "spatial_evidence": [
        {"type": "bbox", "coordinates": [x1, y1, x2, y2], "label": "..."}
    ],
    "artifacts": ["path/to/visualization.png"],
    "metadata": {
        "inference_time_s": 2.3,
        "image": "filename.png"
    },
    "warnings": []
}
```

## Directory Structure

```
SIH/
├── backend/
│   ├── serve.py              # FastAPI model server
│   ├── client.py             # HTTP client for P1
│   └── tools/
│       ├── interfaces.py     # ToolResult contract
│       ├── vqa.py            # run_vqa()
│       ├── caption.py        # run_caption()
│       ├── grounding.py      # run_grounding()
│       └── change.py         # run_change_vqa()
├── models/
│   ├── vqa/
│   │   ├── model.py          # SatQueryVLM singleton
│   │   └── inference.py      # VQA + caption inference
│   ├── grounding/
│   │   └── inference.py      # Bbox parsing + grounding
│   └── change/
│       ├── model.py          # P3 bridge + fallback
│       └── inference.py      # Change region extraction
├── training/
│   ├── bigearthnet/
│   │   ├── prepare.py        # Dataset download + prep
│   │   └── convert.py        # Parquet → LLaVA JSON
│   └── finetuning/
│       ├── config.yaml       # LoRA hyperparameters
│       ├── train.py          # QLoRA training script
│       └── evaluate.py       # Base vs adapted comparison
├── evaluation/
│   ├── metrics.py            # IoU, exact match
│   ├── visualize.py          # Bounding box visualization
│   ├── vrsbench/             # VRSBench benchmark
│   ├── rsvqa/                # RSVQA benchmark
│   └── cdvqa/                # Change Detection VQA
├── configs/                  # Shared configuration
├── datasets/                 # Data (gitignored)
└── docs/                     # Documentation
```
