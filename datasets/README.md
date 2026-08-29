# SatQuery Datasets

## Overview

This directory contains dataset metadata and download scripts.
**Actual data files are gitignored** — do not commit images or large files.

## Datasets Used

### 1. BigEarthNet.txt (Training / Adaptation)
- **Purpose**: Fine-tuning GeoChat-7B → SatQuery-RS
- **Size**: 464K image pairs, 9.6M text annotations
- **Modalities**: Sentinel-1 SAR + Sentinel-2 multispectral
- **Download**: `python -m training.bigearthnet.prepare --subset 100000`
- **Source**: [HuggingFace](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt)

### 2. VRSBench (Evaluation)
- **Purpose**: Benchmark evaluation of VQA, captioning, grounding
- **Size**: 29,614 images, 123,221 VQA pairs
- **Download**: `python -m evaluation.vrsbench.run_vrsbench`
- **Source**: [GitHub](https://github.com/lx709/VRSBench)

### 3. RSVQA (Evaluation)
- **Purpose**: Remote sensing VQA evaluation
- **Variants**: RSVQA-LR (low resolution), RSVQA-HR (high resolution)
- **Download**: Manual from Zenodo
- **Source**: [Project Page](https://rsvqa.sylvainlobry.com/)

### 4. CDVQA (Person 3 — Change Detection VQA)
- Person 3's responsibility

### 5. LEVIR-CD / LEVIR-CC (Person 3 — Change Detection)
- Person 3's responsibility

## Directory Structure

```
datasets/
├── README.md                    # This file
├── bigearthnet/
│   ├── raw/                     # Downloaded parquet (gitignored)
│   ├── images/                  # Raw Sentinel-2 patches (gitignored)
│   ├── rgb/                     # RGB composites (gitignored)
│   └── processed/               # Instruction JSONs (gitignored)
├── vrsbench/
│   └── (evaluation data)
├── rsvqa/
│   └── (evaluation data)
├── cdvqa/                       # Person 3
├── levir_cd/                    # Person 3
└── levir_cc/                    # Person 3
```
