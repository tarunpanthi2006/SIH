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
# STEP 5: Get BigEarthNet images from YOUR HF repo
# ─────────────────────────────────────────────
echo ""
if [ "$DOWNLOAD_IMAGES" = true ]; then
    echo "🛰️  [5/6] Downloading BigEarthNet images from your HF repo..."
    echo "    Repo: $HF_CHECKPOINT_REPO"
    echo ""

    python - <<'EOF'
import os, shutil
from pathlib import Path
from huggingface_hub import snapshot_download

HF_TOKEN = "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"
HF_REPO  = "Sh1vam26/satquery-rs-vlm"
RGB_DEST = Path("datasets/bigearthnet/rgb")
RGB_DEST.mkdir(parents=True, exist_ok=True)

# First check: did step 3 (snapshot_download) already put images somewhere?
checkpoint_dir = Path("models/checkpoints/satquery-rs-vlm")
possible_src_dirs = [
    checkpoint_dir / "rgb",
    checkpoint_dir / "datasets" / "bigearthnet" / "rgb",
    checkpoint_dir / "images",
    checkpoint_dir / "bigearthnet" / "rgb",
    Path("datasets/bigearthnet/images"),
]

moved = 0
for src_dir in possible_src_dirs:
    if src_dir.exists():
        pngs = list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.tif"))
        if pngs:
            print(f"  Found {len(pngs)} images in {src_dir} — moving to datasets/bigearthnet/rgb/")
            for img in pngs:
                dest = RGB_DEST / img.name
                if not dest.exists():
                    shutil.copy2(img, dest)
                    moved += 1
            if moved > 0:
                print(f"  ✅ Moved {moved} images")
                break

if moved > 0:
    print(f"  Total images in rgb/: {len(list(RGB_DEST.glob('*.png')))}")
else:
    # Images not already downloaded — do a targeted download from HF repo
    print(f"  Images not found locally — downloading from {HF_REPO}...")
    print(f"  (This may take 10-30 minutes depending on how many images are stored)")

    try:
        # Download only the rgb/ or datasets/ subfolder from the repo
        dl_dir = snapshot_download(
            repo_id=HF_REPO,
            repo_type="model",
            token=HF_TOKEN,
            local_dir="models/checkpoints/satquery-rs-vlm",
            # Download everything — images should be in there
            ignore_patterns=["*.bin", "*.safetensors", "*.pt", "optimizer*"],
        )
        print(f"  Downloaded to: {dl_dir}")

        # Now search for images in the downloaded folder
        dl_path = Path(dl_dir)
        all_imgs = (list(dl_path.rglob("*.png")) +
                    list(dl_path.rglob("*.jpg")) +
                    list(dl_path.rglob("*.tif")))

        # Filter out any that aren't satellite images (rough heuristic: >10KB)
        sat_imgs = [f for f in all_imgs if f.stat().st_size > 10_000]

        print(f"  Found {len(sat_imgs)} image files in downloaded repo")

        for img in sat_imgs:
            dest = RGB_DEST / img.name
            if not dest.exists():
                shutil.copy2(img, dest)

        final_count = len(list(RGB_DEST.glob("*.png")))
        print(f"  ✅ {final_count} images now in datasets/bigearthnet/rgb/")

        if final_count == 0:
            print("  ⚠️  No images found in your HF repo.")
            print("  Your HF repo may only contain model weights, not images.")
            print("  For --perplexity eval, images are NOT needed. Just run:")
            print("    python -m training.finetuning.evaluate --perplexity --max-samples 100")

    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        import traceback; traceback.print_exc()
EOF

else
    echo "⏭️  [5/6] Skipping image download (--no-images flag set)"
    echo "    For perplexity eval (no images needed):"
    echo "      python -m training.finetuning.evaluate --perplexity --max-samples 100"
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
