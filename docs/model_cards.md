# SatQuery Specialist Layer: Model Cards

## 1. ChangeFormer
- **Task:** Bi-temporal Change Detection
- **Architecture:** MiT-b2 Siamese Encoder + Difference Decoder
- **Input:** Two co-registered RGB images, 3 channels, normalized to [0,1]
- **Output:** Binary change mask (2 classes)
- **Checkpoint:** `ChangeFormer_LEVIR.pth` (trained on building change detection)
- **VRAM:** ~4GB for 256x256 inference. Tiled inference supported for larger images.

## 2. SkySense++
- **Task:** Cross-modal Optical + SAR Analysis (Land Cover)
- **Architecture:** Independent backbones + Shared Transformer Fusion Encoder
- **Optical Input:** 3 channels (RGB), normalized to [0,1]
- **SAR Input:** 2 channels (VV, VH), radar backscatter converted to dB and normalized to [-1, 1]
- **Output:** Scene-level and per-pixel land cover classification (9 classes)
- **VRAM:** 16-24GB (loaded in float16 for efficiency on Lightning AI)

## 3. Prithvi-EO-2.0
- **Task:** Multispectral Analysis
- **Architecture:** ViT-MAE (ibm-nasa-geospatial/Prithvi-EO-2.0-600M)
- **Input:** 6 bands (Blue, Green, Red, NIR, SWIR1, SWIR2), Z-score normalized
- **Output:** Dense feature embeddings
- **VRAM:** ~8-12GB depending on sequence length

## 4. SatQuery-RS (VLM)

### Model Overview

| Field | Value |
|---|---|
| **Model Name** | SatQuery-RS |
| **Base Model** | [GeoChat-7B](https://huggingface.co/MBZUAI/geochat-7B) |
| **Architecture** | LLaVA-1.5 (CLIP ViT-L/14-336 + Vicuna-v1.5 7B) |
| **Adaptation Method** | LoRA (Low-Rank Adaptation) |
| **Training Data** | BigEarthNet.txt (60K–100K instruction samples) |
| **Tasks** | VQA, Captioning, Visual Grounding |
| **Parameters** | 7B total, ~40M trainable (LoRA adapters only) |

### Architecture

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

### Inference Requirements

| Mode | GPU VRAM | System RAM | Speed |
|---|---|---|---|
| 4-bit quantized | 6 GB | 16 GB | ~5–10 sec/query |
| CPU offload | 5 GB | 24 GB | ~30–60 sec/query |
| Full FP16 | 16 GB | 32 GB | ~3–5 sec/query |

### Capabilities
- **Visual Question Answering (VQA)**: Answers natural language questions about satellite images
- **Scene Captioning**: Generates detailed descriptions of remote sensing scenes
- **Visual Grounding**: Locates specific objects/features in satellite imagery
