# BigEarthNet.txt — Data Pipeline Documentation

## Overview

BigEarthNet.txt is a large-scale multi-sensor image-text dataset for Earth observation vision-language learning, containing:
- **464,044** co-registered Sentinel-1 (SAR) + Sentinel-2 (multispectral) image pairs
- **~9.6 million** text annotations (VQA, captions, grounding instructions)
- Geographically anchored metadata (lat/lon, country, season, climate zone)

## Source

| Resource | Location |
|---|---|
| Parquet (text annotations) | [HuggingFace: BIFOLD-BigEarthNetv2-0/BigEarthNet.txt](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt) |
| Sentinel-2 imagery | [Zenodo: BigEarthNet v2.0](https://zenodo.org/records/10891137) |
| Paper | Herzog et al., "BigEarthNet.txt: A Large-Scale Multi-Sensor Image-Text Dataset" (2026) |

## Parquet Schema

| Column | Type | Description |
|---|---|---|
| `ID` | string | Unique sample identifier |
| `s1_name` | string | Sentinel-1 patch name (SAR) |
| `patch_id` | string | Sentinel-2 patch name (multispectral) |
| `input` | string | Instruction/question for VLM |
| `output` | string | Reference answer |
| `type` | string | `binary`, `mcq`, `captioning`, `bounding box` |
| `category` | string | Fine-grained task category |
| `split` | string | `train`, `validation`, `test`, `bench` |
| `latitude` | float | Patch center latitude |
| `longitude` | float | Patch center longitude |
| `country` | string | Acquisition country |
| `season` | string | Acquisition season |
| `climate_zone` | string | Köppen-Geiger zone |

## Pipeline Steps

```bash
# Step 1: Download parquet and inspect
python -m training.bigearthnet.prepare --subset 100000

# Step 2: Download Sentinel-2 imagery (manual from Zenodo for now)

# Step 3: Create RGB composites from Sentinel-2 bands
python -m training.bigearthnet.prepare --create-rgb

# Step 4: Convert to LLaVA instruction format
python -m training.bigearthnet.convert \
    --parquet datasets/bigearthnet/raw/BigEarthNet.txt.parquet \
    --images-dir datasets/bigearthnet/rgb \
    --output-dir datasets/bigearthnet/processed \
    --max-samples 100000

# Step 5: Validate converted data
python -m training.bigearthnet.validate \
    --data datasets/bigearthnet/processed/bigearthnet_instructions.json

# Step 6: Create train/val/test splits
python -m training.bigearthnet.splits \
    --data datasets/bigearthnet/processed/bigearthnet_instructions.json \
    --output-dir datasets/bigearthnet/processed
```

## Output Format (LLaVA Instruction JSON)

```json
{
    "id": "ben_binary_00001",
    "image": "datasets/bigearthnet/rgb/S2A_..._patch.png",
    "conversations": [
        {"from": "human", "value": "<image>\nIs there a water body present?"},
        {"from": "gpt", "value": "Yes"}
    ],
    "metadata": {
        "source": "bigearthnet_txt",
        "task_type": "binary",
        "category": "presence",
        "patch_id": "S2A_..._patch",
        "split": "train"
    }
}
```

## Directory Structure After Processing

```
datasets/bigearthnet/
├── raw/
│   └── BigEarthNet.txt.parquet          # Original parquet
├── images/
│   └── S2A_MSIL2A_.../                  # Raw Sentinel-2 patches
│       ├── ..._B02.tif
│       ├── ..._B03.tif
│       └── ..._B04.tif
├── rgb/
│   └── S2A_MSIL2A_..._patch.png         # RGB composites
├── processed/
│   ├── bigearthnet_instructions.json     # Full converted data
│   ├── train.json                        # Training split
│   ├── validation.json                   # Validation split
│   ├── test.json                         # Test split
│   ├── debug.json                        # 1K debug subset
│   ├── train_small.json                  # 10K small training set
│   └── val_small.json                    # 500 validation samples
└── selected_patches.txt                  # Selected patch IDs
```
