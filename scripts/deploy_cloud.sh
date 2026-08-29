#!/bin/bash
# ============================================================
# SatQuery-RS — Cloud GPU Deployment Script
# ============================================================
# Run this on a cloud GPU (JarvisLabs, RunPod, etc.) to:
# 1. Install dependencies
# 2. Download model weights
# 3. Start the API server
#
# Usage:
#   bash scripts/deploy_cloud.sh
#
# Pre-requisites:
#   - Set HF_TOKEN environment variable
#   - Set HF_HUB_REPO for your LoRA adapter
# ============================================================

set -e

echo "============================================"
echo "  SatQuery-RS Cloud Deployment"
echo "============================================"

# ---- Step 1: Install dependencies ----
echo ""
echo "[1/5] Installing Python dependencies..."
pip install -q -r requirements.txt

# ---- Step 2: Login to HuggingFace ----
echo ""
echo "[2/5] Setting up HuggingFace..."
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    echo "  ✅ HuggingFace login successful"
else
    echo "  ⚠️  HF_TOKEN not set. Set it: export HF_TOKEN=your_token"
    echo "      Get token at: https://huggingface.co/settings/tokens"
fi

# ---- Step 3: Download base model ----
echo ""
echo "[3/5] Downloading GeoChat-7B base model (~14 GB)..."
python scripts/download_models.py --base

# ---- Step 4: Download LoRA adapter ----
echo ""
echo "[4/5] Downloading SatQuery-RS LoRA adapter..."
if [ -n "$HF_HUB_REPO" ]; then
    python scripts/download_models.py --adapter --repo "$HF_HUB_REPO"
    echo "  ✅ Adapter downloaded"
else
    echo "  ⚠️  HF_HUB_REPO not set. Using base model without adapter."
    echo "      Set it: export HF_HUB_REPO=your-username/satquery-rs-vlm"
fi

# ---- Step 5: Start API server ----
echo ""
echo "[5/5] Starting SatQuery-RS API server..."
echo ""
echo "============================================"
echo "  Server starting on port 8100"
echo "  API docs: http://0.0.0.0:8100/docs"
echo "  Health:   http://0.0.0.0:8100/health"
echo "============================================"
echo ""

# Use quantized-4bit mode on cloud GPU (fast, plenty of VRAM)
export SATQUERY_MODE=quantized-4bit
python -m backend.serve --host 0.0.0.0 --port 8100
