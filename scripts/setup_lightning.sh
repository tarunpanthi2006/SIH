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
# STEP 5: Download & extract BigEarthNet images from YOUR HF repo
# ─────────────────────────────────────────────
echo ""
if [ "$DOWNLOAD_IMAGES" = true ]; then
    echo "🛰️  [5/6] Downloading BigEarthNet images (bigearthnet_extracted.tar.gz)..."
    echo "    Source: Sh1vam26/satquery-rs-vlm (10.2 GB)"
    echo "    This will take 10-20 minutes depending on your connection speed"
    echo ""

    mkdir -p datasets/bigearthnet/rgb

    # Check if already extracted
    IMG_COUNT=$(ls datasets/bigearthnet/rgb/*.png 2>/dev/null | wc -l || echo 0)
    if [ "$IMG_COUNT" -gt "100" ]; then
        echo "  ✅ Images already extracted ($IMG_COUNT images found) — skipping download"
    else
        python - <<'EOF'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

HF_TOKEN = "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"
HF_REPO  = "Sh1vam26/satquery-rs-vlm"
TAR_FILE = "bigearthnet_extracted.tar.gz"
TAR_DEST = Path("datasets/bigearthnet") / TAR_FILE

# Step 1: Download the tar.gz if not already present
if TAR_DEST.exists() and TAR_DEST.stat().st_size > 1_000_000:
    print(f"  ✅ {TAR_FILE} already downloaded ({TAR_DEST.stat().st_size / 1e9:.1f} GB)")
else:
    print(f"  Downloading {TAR_FILE} from {HF_REPO}...")
    print(f"  File size: ~10.2 GB — please be patient!")
    TAR_DEST.parent.mkdir(parents=True, exist_ok=True)

    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO,
            filename=TAR_FILE,
            repo_type="model",
            token=HF_TOKEN,
            local_dir=str(TAR_DEST.parent),
            local_dir_use_symlinks=False,
        )
        print(f"  ✅ Downloaded to: {downloaded}")
        TAR_DEST = Path(downloaded)
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        import traceback; traceback.print_exc()
        exit(1)
EOF

    # Step 2: Extract the tar.gz using bash (much faster and no memory limits)
    echo ""
    echo "  Extracting bigearthnet_extracted.tar.gz..."
    mkdir -p datasets/bigearthnet/_extracted_tmp
    
    # Extract using tar directly
    tar -xzf datasets/bigearthnet/bigearthnet_extracted.tar.gz -C datasets/bigearthnet/_extracted_tmp/
    echo "  Extraction complete!"

    # Step 3: Move files to the correct locations
    echo "  Moving images to datasets/bigearthnet/rgb/..."
    
    # The tar might contain datasets/bigearthnet/rgb/ or just rgb/ or just files.
    # We use find to efficiently locate all image files and move them.
    find datasets/bigearthnet/_extracted_tmp/ -type f \( -iname \*.png -o -iname \*.jpg -o -iname \*.tif -o -iname \*.tiff \) -exec mv -n {} datasets/bigearthnet/rgb/ \;
    
    # Cleanup tmp dir
    rm -rf datasets/bigearthnet/_extracted_tmp/

    FINAL_COUNT=$(ls datasets/bigearthnet/rgb/*.png 2>/dev/null | wc -l || echo 0)
    echo "  ✅ Done! $FINAL_COUNT images now in datasets/bigearthnet/rgb/"
    
    fi

else
    echo "⏭️  [5/6] Skipping image download (--no-images flag set)"
    echo "    Run perplexity eval without images:"
    echo "      python -m training.finetuning.evaluate --perplexity --max-samples 100"
fi

# ─────────────────────────────────────────────
# MAPPING VERIFICATION — confirm text↔image alignment
# ─────────────────────────────────────────────
echo ""
echo "🔍 Verifying text ↔ image mapping..."
python - <<'EOF'
import json
from pathlib import Path

val_path = Path("datasets/bigearthnet/processed/val_small.json")
rgb_dir  = Path("datasets/bigearthnet/rgb")

if not val_path.exists():
    print("  ⚠️  val_small.json not found — skipping verification")
    exit(0)

with open(val_path) as f:
    data = json.load(f)

# Check how many val samples have a matching image on disk
matched   = 0
missing   = 0
no_path   = 0
examples_missing = []

for sample in data:
    img_path = sample.get("image", "")
    if not img_path:
        no_path += 1
        continue

    img_file = Path(img_path)
    if img_file.exists():
        matched += 1
    else:
        missing += 1
        if len(examples_missing) < 3:
            examples_missing.append(img_path)

total = len(data)
match_pct = matched / max(total, 1) * 100

print(f"  Total val samples : {total}")
print(f"  ✅ Matched (image exists): {matched}  ({match_pct:.1f}%)")
print(f"  ❌ Missing images        : {missing}")
print(f"  ⚪ No image path         : {no_path}")

if examples_missing:
    print(f"\n  Example missing paths:")
    for p in examples_missing:
        print(f"    {p}")
    # Show what IS in the rgb dir for comparison
    sample_imgs = list(rgb_dir.glob("*.png"))[:3] if rgb_dir.exists() else []
    if sample_imgs:
        print(f"\n  Example images actually on disk:")
        for img in sample_imgs:
            print(f"    {img}")

if match_pct >= 80:
    print(f"\n  ✅ MAPPING IS GOOD — {match_pct:.0f}% of val samples have matching images!")
    print(f"     Ready to run: python -m training.finetuning.evaluate --compare --max-samples 50")
elif match_pct >= 30:
    print(f"\n  ⚠️  PARTIAL MAPPING — only {match_pct:.0f}% matched.")
    print(f"     The val split may use patches not in the tar.gz (which has train split images).")
    print(f"     Eval will still work but on fewer samples.")
else:
    print(f"\n  ❌ MAPPING FAILED — images downloaded but patch IDs don't match!")
    print(f"     The filenames in the tar.gz don't match the patch_ids in val_small.json.")
    print(f"     Run perplexity eval instead (no images needed):")
    print(f"       python -m training.finetuning.evaluate --perplexity --max-samples 100")
EOF

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
