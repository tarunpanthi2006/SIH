# SatQuery — Development Guide (Person 2: VLM)

## Quick Start

### 1. Environment Setup

```bash
# Clone the repo
git clone <repo-url>
cd SIH
git checkout feature/vlm

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Copy environment config
copy .env.example .env
# Edit .env with your HuggingFace token
```

### 2. Download Model Weights

```bash
# Login to HuggingFace (one-time)
huggingface-cli login

# Download GeoChat-7B base model (~14 GB)
python scripts/download_models.py --base

# Check download status
python scripts/download_models.py --status
```

### 3. Run Tests (No GPU Needed)

```bash
# Run all schema and unit tests
pytest tests/ -v

# Run specific test
pytest tests/test_vqa.py -v
pytest tests/test_data_pipeline.py -v
```

### 4. Test Inference (Requires GPU)

```bash
# Test with a sample image
python -m backend.tools.vqa --image path/to/satellite.png --question "What is visible?"
python -m backend.tools.caption --image path/to/satellite.png
python -m backend.tools.grounding --image path/to/satellite.png --query "water body"

# Schema-only test (no GPU needed)
python -m backend.tools.vqa --test
python -m backend.tools.caption --test
python -m backend.tools.grounding --test
```

## Data Pipeline

### Prepare BigEarthNet.txt

```bash
# Step 1: Download and inspect parquet
python -m training.bigearthnet.prepare --subset 100000

# Step 2: Convert to instruction format (60K-100K samples)
python -m training.bigearthnet.convert \
    --parquet datasets/bigearthnet/raw/BigEarthNet.txt.parquet \
    --output-dir datasets/bigearthnet/processed \
    --max-samples 100000

# Step 3: Validate
python -m training.bigearthnet.validate \
    --data datasets/bigearthnet/processed/bigearthnet_instructions.json

# Step 4: Create splits
python -m training.bigearthnet.splits \
    --data datasets/bigearthnet/processed/bigearthnet_instructions.json \
    --output-dir datasets/bigearthnet/processed
```

## Training on Cloud GPU (H100)

### Setup Script (Run at Start of Each Session)

```bash
# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace
huggingface-cli login --token $HF_TOKEN

# Download base model
python scripts/download_models.py --base

# If resuming: download previous checkpoint
python scripts/download_models.py --adapter --repo your-username/satquery-rs-vlm
```

### Training Commands

```bash
# Debug run first (1K samples, ~5 minutes)
python -m training.finetuning.train --config training/finetuning/config.yaml --debug

# Small run (10K samples, ~45 minutes)
python -m training.finetuning.train --config training/finetuning/config.yaml --small

# Full run (all training data)
python -m training.finetuning.train --config training/finetuning/config.yaml

# Resume from Hub (when switching cloud accounts)
python -m training.finetuning.train --config training/finetuning/config.yaml --resume-from-hub
```

### Multi-Account Training Workflow

1. **Account 1**: Setup + debug run + start training → checkpoint auto-pushed to HF Hub
2. **Account 2**: Setup + `--resume-from-hub` → continues training → push checkpoint
3. **Account 3**: Same → continues training → final adapter saved

## Local Inference (RTX 4050 6GB)

```bash
# Set inference mode in .env
SATQUERY_MODE=cpu-offload   # Safe mode (recommended for 6GB)
# SATQUERY_MODE=quantized-4bit  # Fast but tight on 6GB

# Test inference
python -m backend.tools.vqa --image test_satellite.png --question "What land cover is present?"
```

## Evaluation

```bash
# Base vs Adapted comparison
python -m training.finetuning.evaluate \
    --test-data datasets/bigearthnet/processed/test.json \
    --compare \
    --max-samples 200

# VRSBench evaluation
python -m evaluation.vrsbench.run_vrsbench --max-samples 500

# RSVQA evaluation
python -m evaluation.rsvqa.run_rsvqa --data <path-to-rsvqa-json> --max-samples 500
```

## Project Structure (Person 2 Files)

```
training/
├── bigearthnet/
│   ├── prepare.py         # Download + organize dataset
│   ├── convert.py         # Parquet → LLaVA JSON
│   ├── validate.py        # Data integrity checks
│   ├── splits.py          # Train/val/test splits
│   └── README.md          # Data documentation
├── finetuning/
│   ├── config.yaml        # LoRA hyperparameters
│   ├── train.py           # Training script
│   └── evaluate.py        # Base vs adapted evaluation

models/
├── vqa/
│   ├── model.py           # SatQueryVLM loader (singleton)
│   └── inference.py       # VQA + caption inference
├── grounding/
│   └── inference.py       # Grounding + bbox parsing
└── checkpoints/
    └── satquery-rs-vlm/   # LoRA adapter weights (gitignored)

backend/tools/
├── interfaces.py          # ToolResult contract
├── vqa.py                 # run_vqa() — Person 1 interface
├── caption.py             # run_caption() — Person 1 interface
└── grounding.py           # run_grounding() — Person 1 interface
```

## Output Contract

Every tool returns:
```json
{
    "task": "vqa|caption|grounding",
    "model": "SatQuery-RS",
    "answer": "...",
    "confidence": 0.0-1.0,
    "spatial_evidence": [],
    "artifacts": [],
    "metadata": {},
    "warnings": []
}
```
