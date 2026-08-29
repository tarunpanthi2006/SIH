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
