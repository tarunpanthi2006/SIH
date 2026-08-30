#!/bin/bash
# ============================================================
# SatQuery-RS — Complete Lightning AI Setup Script
# ============================================================
# This script handles EVERYTHING in one shot:
#   1. Git pull latest code
#   2. Install all dependencies
#   3. Download LoRA checkpoint from HuggingFace Hub
#   4. Download evaluation JSON data
#   5. Download BigEarthNet images from HuggingFace (for full eval)
#   6. Configure environment
#
# Usage:
#   bash scripts/setup_lightning.sh          # Full setup with images
#   bash scripts/setup_lightning.sh --no-images  # Skip image download
# ============================================================

set -e  # Exit immediately on any error

DOWNLOAD_IMAGES=true
if [[ "$1" == "--no-images" ]]; then
    DOWNLOAD_IMAGES=false
fi

# ── Config ──────────────────────────────────────────────────
HF_TOKEN="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"
HF_CHECKPOINT_REPO="Sh1vam26/satquery-rs-vlm"
HF_IMAGES_REPO="danielz01/BigEarthNet-S2-v1.0"
PROJECT_DIR="/teamspace/studios/this_studio/SIH"
# ────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     SatQuery-RS — Full Lightning AI Environment Setup       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────
# STEP 1: Clone or pull repo
# ─────────────────────────────────────────────
echo "🔄 [1/6] Getting latest code..."
if [ ! -d "$PROJECT_DIR" ]; then
    cd /teamspace/studios/this_studio
    git clone https://github.com/tarunpanthi2006/SIH.git
    cd SIH
    git checkout feature/vlm
else
    cd "$PROJECT_DIR"
    git fetch origin feature/vlm
    git checkout feature/vlm
    git pull origin feature/vlm
fi
echo "✅ Code ready at: $PROJECT_DIR"

# ─────────────────────────────────────────────
# STEP 2: Install Python dependencies
# ─────────────────────────────────────────────
echo ""
echo "📦 [2/6] Installing Python dependencies..."
pip install -q -r requirements.txt
# Critical: pin accelerate for transformers 4.36.x compatibility
pip install -q "accelerate==0.27.2"
# Install rasterio for reading Sentinel-2 .tif files
pip install -q rasterio 2>/dev/null || echo "  (rasterio optional — skipping)"
echo "✅ All dependencies installed"

# ─────────────────────────────────────────────
# STEP 3: Download LoRA Checkpoint from Hub
# ─────────────────────────────────────────────
echo ""
echo "🤖 [3/6] Downloading LoRA checkpoint (checkpoint-1250)..."
mkdir -p models/checkpoints/satquery-rs-vlm

python - <<EOF
import os
from huggingface_hub import snapshot_download

print("  Downloading from: $HF_CHECKPOINT_REPO")
print("  This may take 2-5 minutes...")
snapshot_download(
    repo_id="$HF_CHECKPOINT_REPO",
    local_dir="models/checkpoints/satquery-rs-vlm",
    repo_type="model",
    token="$HF_TOKEN",
)
print("  Download complete!")
EOF

# Fix nested checkpoints/ subfolder if Hub created one
if [ -d "models/checkpoints/satquery-rs-vlm/checkpoints" ]; then
    echo "  📁 Fixing nested folder structure..."
    mv models/checkpoints/satquery-rs-vlm/checkpoints/checkpoint-* \
       models/checkpoints/satquery-rs-vlm/ 2>/dev/null || true
    rmdir models/checkpoints/satquery-rs-vlm/checkpoints 2>/dev/null || true
fi

echo "✅ Checkpoint ready. Available:"
ls -d models/checkpoints/satquery-rs-vlm/checkpoint-* 2>/dev/null || echo "  ⚠️  No checkpoint-* dirs found"

# ─────────────────────────────────────────────
# STEP 4: Get evaluation JSON data
# ─────────────────────────────────────────────
echo ""
echo "📊 [4/6] Preparing evaluation data..."

mkdir -p datasets/bigearthnet/processed

if [ ! -f "datasets/bigearthnet/processed/val_small.json" ]; then
    echo "  val_small.json not found — generating from BigEarthNet.txt parquet..."
    python - <<'EOF'
import json
from pathlib import Path

token = "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"

# Step A: Download BigEarthNet.txt parquet from HuggingFace Datasets
parquet_path = Path("datasets/bigearthnet/raw/BigEarthNet.txt.parquet")
if not parquet_path.exists():
    print("  Downloading BigEarthNet.txt parquet from HuggingFace...")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    try:
        dl = hf_hub_download(
            repo_id="BIFOLD-BigEarthNetv2-0/BigEarthNet.txt",
            filename="BigEarthNet.txt.parquet",
            repo_type="dataset",
            local_dir=str(parquet_path.parent),
            token=token,
        )
        parquet_path = Path(dl)
        print(f"  Parquet downloaded: {parquet_path}")
    except Exception as e:
        print(f"  ❌ Parquet download failed: {e}")
        exit(1)
else:
    print(f"  Parquet already exists: {parquet_path}")

# Step B: Generate val_small.json from parquet val split
import pandas as pd
print("  Reading parquet and extracting val split...")
df = pd.read_parquet(parquet_path)
print(f"  Total rows: {len(df):,}, columns: {list(df.columns)}")

# Handle different column naming conventions
split_col  = next((c for c in df.columns if "split" in c.lower()), None)
input_col  = next((c for c in df.columns if c.lower() in ("input", "question", "instruction")), None)
output_col = next((c for c in df.columns if c.lower() in ("output", "answer", "response")), None)
patch_col  = next((c for c in df.columns if "patch" in c.lower()), None)
type_col   = next((c for c in df.columns if c.lower() in ("type", "task_type", "category")), None)

if split_col:
    val_df = df[df[split_col] == "val"].head(500)
else:
    val_df = df.head(500)

print(f"  Val samples: {len(val_df)}")

samples = []
for _, row in val_df.iterrows():
    patch_id = str(row.get(patch_col, f"sample_{len(samples)}")) if patch_col else f"sample_{len(samples)}"
    inp  = str(row.get(input_col,  "Describe this satellite image.")) if input_col  else "Describe this satellite image."
    out  = str(row.get(output_col, "")) if output_col else ""
    typ  = str(row.get(type_col,   "captioning")) if type_col else "captioning"
    samples.append({
        "id": patch_id,
        "image": f"datasets/bigearthnet/rgb/{patch_id}.png",
        "conversations": [
            {"from": "human", "value": f"<image>\n{inp}"},
            {"from": "gpt",   "value": out},
        ],
        "metadata": {"task_type": typ, "patch_id": patch_id}
    })

out_path = Path("datasets/bigearthnet/processed/val_small.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(samples, f, indent=2)
print(f"  ✅ val_small.json created with {len(samples)} samples → {out_path}")
EOF
else
    COUNT=$(python -c "import json; d=json.load(open('datasets/bigearthnet/processed/val_small.json')); print(len(d))")
    echo "  ✅ val_small.json already exists ($COUNT samples)"
fi

echo "✅ Evaluation data ready"

# ─────────────────────────────────────────────
# STEP 5: Download BigEarthNet images (for full eval)
# ─────────────────────────────────────────────
echo ""
if [ "$DOWNLOAD_IMAGES" = true ]; then
    echo "🛰️  [5/6] Downloading BigEarthNet satellite images..."
    echo "    Source: HuggingFace ($HF_IMAGES_REPO)"
    echo "    This streams only the images referenced in val_small.json"
    echo "    Estimated time: 15-30 minutes depending on GPU machine speed"
    echo ""

    mkdir -p datasets/bigearthnet/rgb

    python - <<'EOF'
import json, os
from pathlib import Path

try:
    import datasets as hf_datasets
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
    import datasets as hf_datasets

from PIL import Image

rgb_dir = Path("datasets/bigearthnet/rgb")
val_path = Path("datasets/bigearthnet/processed/val_small.json")

if not val_path.exists():
    print("  val_small.json not found — skipping image download")
    exit(0)

with open(val_path) as f:
    data = json.load(f)

# Get all unique patch IDs we need
needed = set()
for item in data:
    img_path = item.get("image", "")
    if img_path:
        patch_id = Path(img_path).stem
        if patch_id and patch_id != "":
            needed.add(patch_id)

# Also check bigearthnet_instructions.json if it exists
inst_path = Path("datasets/bigearthnet/processed/bigearthnet_instructions.json")
if inst_path.exists():
    with open(inst_path) as f:
        inst_data = json.load(f)
    for item in inst_data:
        pid = item.get("metadata", {}).get("patch_id", "")
        if pid:
            needed.add(pid)

# Filter out already downloaded
existing = set(f.stem for f in rgb_dir.glob("*.png"))
needed = needed - existing

print(f"  Need {len(needed)} images, already have {len(existing)}")

if len(needed) == 0:
    print("  All images already downloaded!")
    exit(0)

print(f"  Streaming {len(needed)} images from HuggingFace CDN...")
print("  (Progress shown every 100 images)")

HF_TOKEN = "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"

try:
    ds = hf_datasets.load_dataset(
        "danielz01/BigEarthNet-S2-v1.0",
        "s2-rgb",
        split="train",
        streaming=True,
        token=HF_TOKEN,
    ).cast_column("img", hf_datasets.Image(decode=False))

    saved = 0
    for item in ds:
        img_data = item["img"]
        patch_id = Path(img_data["path"]).stem

        if patch_id in needed:
            out = rgb_dir / f"{patch_id}.png"
            with open(out, "wb") as f:
                f.write(img_data["bytes"])
            needed.remove(patch_id)
            saved += 1

            if saved % 100 == 0:
                print(f"  Saved {saved} images... {len(needed)} remaining")

            if len(needed) == 0:
                break

    print(f"  ✅ Downloaded {saved} images to datasets/bigearthnet/rgb/")
    if needed:
        print(f"  ⚠️  {len(needed)} images not found in dataset (may be in test split)")

except Exception as e:
    print(f"  ❌ Image download failed: {e}")
    print("  You can still run --perplexity eval without images")
EOF

else
    echo "⏭️  [5/6] Skipping image download (--no-images flag set)"
    echo "    Run with full eval using: bash scripts/setup_lightning.sh"
    echo "    Or download images separately: python training/bigearthnet/fast_image_download.py"
fi

# ─────────────────────────────────────────────
# STEP 6: Write .env and final checks
# ─────────────────────────────────────────────
echo ""
echo "🔧 [6/6] Configuring environment..."

# Write .env for the project
cat > .env <<ENVEOF
# ============================================================
# SatQuery Environment Configuration (auto-generated)
# ============================================================

# ----- HuggingFace -----
HF_TOKEN=$HF_TOKEN
HF_HUB_REPO=$HF_CHECKPOINT_REPO

# ----- Model Paths -----
GEOCHAT_BASE_MODEL=MBZUAI/geochat-7B
LORA_ADAPTER_PATH=models/checkpoints/satquery-rs-vlm/checkpoint-1250
VISION_TOWER=openai/clip-vit-large-patch14-336

# ----- Inference Mode -----
SATQUERY_MODE=quantized-4bit

# ----- Device -----
DEVICE_MAP=auto

# ----- Paths -----
DATASET_ROOT=datasets
CHECKPOINT_DIR=models/checkpoints
ENVEOF

echo "✅ .env written"

# ── Final Summary ────────────────────────────────────────────
IMAGE_COUNT=$(ls datasets/bigearthnet/rgb/*.png 2>/dev/null | wc -l || echo 0)

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  ✅ SETUP COMPLETE!                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  📁 Checkpoint:      models/checkpoints/satquery-rs-vlm/checkpoint-1250"
echo "  📊 Eval data:       datasets/bigearthnet/processed/val_small.json"
echo "  🛰️  Images on disk:  $IMAGE_COUNT images"
echo ""
echo "  ─────────────────────────────────────────────────────────"
echo "  🔬 Run perplexity eval (NO images needed, ~10 min):"
echo "     python -m training.finetuning.evaluate --perplexity --max-samples 100"
echo ""
echo "  🔬 Run FULL comparison — base vs adapted (needs images):"
echo "     python -m training.finetuning.evaluate --compare --max-samples 50"
echo ""
echo "  🔬 Run adapted model only:"
echo "     python -m training.finetuning.evaluate --adapted-only --max-samples 100"
echo ""
echo "  🚀 Resume training from checkpoint-1250:"
echo "     python -m training.finetuning.train --config training/finetuning/config.yaml --resume-from-hub"
echo ""
echo "  🌐 Launch API server:"
echo "     python -m backend.serve --host 0.0.0.0 --port 8080"
echo "  ─────────────────────────────────────────────────────────"
echo ""
