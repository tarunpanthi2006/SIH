#!/bin/bash
# ============================================================
# SatQuery-RS — Complete Lightning AI Setup Script
# ============================================================
# Run this ONCE on a fresh Lightning AI studio to set everything up.
# Usage: bash scripts/setup_lightning.sh
# ============================================================

set -e  # Exit immediately on any error

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       SatQuery-RS — Lightning AI Environment Setup          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────
# STEP 1: Navigate to project root
# ─────────────────────────────────────────────
echo "📁 [1/7] Navigating to project..."
cd /teamspace/studios/this_studio/SIH

# ─────────────────────────────────────────────
# STEP 2: Pull latest code
# ─────────────────────────────────────────────
echo ""
echo "🔄 [2/7] Pulling latest code from GitHub..."
git pull origin feature/vlm
echo "✅ Code up to date"

# ─────────────────────────────────────────────
# STEP 3: Install Python dependencies
# ─────────────────────────────────────────────
echo ""
echo "📦 [3/7] Installing Python dependencies..."
pip install -q -r requirements.txt
# Pin accelerate for transformers 4.36.x compatibility
pip install -q "accelerate==0.27.2"
echo "✅ Dependencies installed"

# ─────────────────────────────────────────────
# STEP 4: Download LoRA Checkpoint from Hub
# ─────────────────────────────────────────────
echo ""
echo "🤖 [4/7] Downloading LoRA checkpoint from HuggingFace Hub..."
echo "    Repo: Sh1vam26/satquery-rs-vlm"
echo "    This downloads checkpoint-1250 (the 0.6-loss model)"
echo ""

mkdir -p models/checkpoints/satquery-rs-vlm

python - <<'EOF'
import os
from huggingface_hub import snapshot_download

token = os.getenv("HF_TOKEN", "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")

print("  Downloading... (may take 2-5 minutes)")
snapshot_download(
    repo_id="Sh1vam26/satquery-rs-vlm",
    local_dir="models/checkpoints/satquery-rs-vlm",
    repo_type="model",
    token=token,
)
print("  ✅ Download complete!")
EOF

# Fix nested checkpoints/ subfolder if Hub created one
if [ -d "models/checkpoints/satquery-rs-vlm/checkpoints" ]; then
    echo "  📁 Fixing nested folder structure..."
    mv models/checkpoints/satquery-rs-vlm/checkpoints/checkpoint-* \
       models/checkpoints/satquery-rs-vlm/ 2>/dev/null || true
    rmdir models/checkpoints/satquery-rs-vlm/checkpoints 2>/dev/null || true
fi

echo "✅ Checkpoint downloaded"
echo ""
echo "  Available checkpoints:"
ls -d models/checkpoints/satquery-rs-vlm/checkpoint-* 2>/dev/null \
    || echo "  (no checkpoint-* dirs found, check download above)"

# ─────────────────────────────────────────────
# STEP 5: Download val_small.json (evaluation data)
# ─────────────────────────────────────────────
echo ""
echo "📊 [5/7] Checking evaluation data..."

VAL_PATH="datasets/bigearthnet/processed/val_small.json"
if [ -f "$VAL_PATH" ]; then
    COUNT=$(python -c "import json; d=json.load(open('$VAL_PATH')); print(len(d))")
    echo "✅ val_small.json already exists ($COUNT samples)"
else
    echo "  val_small.json not found — generating a text-only mini eval set..."
    mkdir -p datasets/bigearthnet/processed

    python - <<'EOF'
import json, random, os

# Try to read from train.json if it exists, otherwise create dummy samples
train_path = "datasets/bigearthnet/processed/train.json"
val_path = "datasets/bigearthnet/processed/val_small.json"

if os.path.exists(train_path):
    with open(train_path) as f:
        data = json.load(f)
    random.seed(42)
    val = random.sample(data, min(200, len(data)))
    with open(val_path, "w") as f:
        json.dump(val, f, indent=2)
    print(f"  Created val_small.json from train.json ({len(val)} samples)")
else:
    # Create minimal dummy eval set for perplexity testing
    # (no images needed for --perplexity mode)
    dummy = [
        {
            "id": f"dummy_{i}",
            "image": "",
            "conversations": [
                {"from": "human", "value": f"What land cover type is visible in this satellite image?"},
                {"from": "gpt", "value": random.choice([
                    "This satellite image shows urban residential areas with mixed vegetation.",
                    "The image depicts agricultural land with crop fields and irrigation patterns.",
                    "Dense forest cover with deciduous trees is visible in this image.",
                    "Industrial zones with buildings and paved surfaces dominate this scene.",
                    "Wetlands and water bodies are visible with surrounding vegetation.",
                ])}
            ],
            "metadata": {"task_type": "captioning"}
        }
        for i in range(100)
    ]
    with open(val_path, "w") as f:
        json.dump(dummy, f, indent=2)
    print(f"  Created minimal dummy val_small.json (100 text-only samples)")
    print(f"  ⚠️  For real evaluation, you need the actual processed/ dataset")
EOF
fi

# ─────────────────────────────────────────────
# STEP 6: Download BigEarthNet images (optional, for generation eval)
# ─────────────────────────────────────────────
echo ""
echo "🛰️  [6/7] Image data setup..."
echo ""
echo "  For --perplexity eval: ❌ NO IMAGES NEEDED → skip this step"
echo "  For --compare eval:    ✅ Images needed"
echo ""
echo "  If you need images, run:"
echo "    python training/bigearthnet/fast_image_download.py --subset val"
echo "  (takes ~20 min for the val subset)"

# ─────────────────────────────────────────────
# STEP 7: Set environment variables
# ─────────────────────────────────────────────
echo ""
echo "🔧 [7/7] Configuring environment..."
export HF_TOKEN="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"
export HF_HUB_REPO="Sh1vam26/satquery-rs-vlm"
export GEOCHAT_BASE_MODEL="MBZUAI/geochat-7B"
export LORA_ADAPTER_PATH="models/checkpoints/satquery-rs-vlm/checkpoint-1250"
export VISION_TOWER="openai/clip-vit-large-patch14-336"
export SATQUERY_MODE="quantized-4bit"
export DEVICE_MAP="auto"
echo "✅ Environment configured"

# ─────────────────────────────────────────────
# DONE — print next steps
# ─────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    ✅ SETUP COMPLETE!                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  🔬 Run perplexity eval (NO images needed, ~10 min):"
echo "     python -m training.finetuning.evaluate --perplexity --max-samples 100"
echo ""
echo "  🔬 Run full comparison (needs images, ~30 min):"
echo "     python -m training.finetuning.evaluate --compare --max-samples 50"
echo ""
echo "  🚀 Resume training from checkpoint-1250:"
echo "     python -m training.finetuning.train --config training/finetuning/config.yaml --resume-from-hub"
echo ""
echo "  🌐 Launch API server (port 8080):"
echo "     SATQUERY_MODE=quantized-4bit python -m backend.serve --host 0.0.0.0 --port 8080"
echo ""
