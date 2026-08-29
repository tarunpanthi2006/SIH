# SatQuery-RS — Model Card

## Model Overview

| Field | Value |
|---|---|
| **Model Name** | SatQuery-RS |
| **Base Model** | [GeoChat-7B](https://huggingface.co/MBZUAI/geochat-7B) |
| **Architecture** | LLaVA-1.5 (CLIP ViT-L/14-336 + Vicuna-v1.5 7B) |
| **Adaptation Method** | LoRA (Low-Rank Adaptation) |
| **Training Data** | BigEarthNet.txt (60K–100K instruction samples) |
| **Tasks** | VQA, Captioning, Visual Grounding |
| **Parameters** | 7B total, ~40M trainable (LoRA adapters only) |

## Architecture

```
Input Image (satellite/RS)
       ↓
CLIP ViT-L/14 @ 336px  (frozen)
       ↓
MLP Projection Layer    (frozen or partially unfrozen)
       ↓
Vicuna-v1.5 7B + LoRA   (LoRA adapters trainable)
       ↓
Generated Text (answer / caption / coordinates)
```

## Training Details

| Hyperparameter | Value |
|---|---|
| LoRA rank (r) | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, v_proj, k_proj, o_proj |
| Quantization | QLoRA NF4 |
| Learning rate | 2e-4 |
| Scheduler | Cosine |
| Batch size (effective) | 32 (4 × 8 gradient accumulation) |
| Epochs | 3 |
| Training GPU | 1× H100 80GB |

## Inference Requirements

| Mode | GPU VRAM | System RAM | Speed |
|---|---|---|---|
| 4-bit quantized | 6 GB | 16 GB | ~5–10 sec/query |
| CPU offload | 5 GB | 24 GB | ~30–60 sec/query |
| Full FP16 | 16 GB | 32 GB | ~3–5 sec/query |

## Capabilities

### 1. Visual Question Answering (VQA)
- Answers natural language questions about satellite images
- Supports: land cover, presence, counting, comparison questions

### 2. Scene Captioning
- Generates detailed descriptions of remote sensing scenes
- Identifies land cover types, structures, and spatial relationships

### 3. Visual Grounding
- Locates specific objects/features in satellite imagery
- Returns bounding box coordinates [x1, y1, x2, y2] normalized to [0, 1]

## Limitations

- Trained primarily on optical (Sentinel-2) imagery; SAR interpretation may be limited
- Grounding accuracy depends on object size relative to image resolution
- Not designed for pixel-level segmentation (use ChangeFormer for that)
- Performance degrades on imagery very different from Sentinel-2 characteristics

## License

Base model (GeoChat-7B): Check [MBZUAI license](https://huggingface.co/MBZUAI/geochat-7B)
LoRA adapter (SatQuery-RS): Project-specific (SIH 2026)
