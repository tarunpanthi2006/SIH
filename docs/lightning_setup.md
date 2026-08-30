# SatQuery-RS — Complete Lightning AI Setup Guide

> Copy-paste these commands one section at a time in your Lightning AI terminal.

---

## 🧭 Overview

| What you need | Time | Required for |
|---|---|---|
| Code (git clone) | ~30 sec | Everything |
| pip install | ~3 min | Everything |
| Checkpoint (HF Hub) | ~3 min | Eval + Serving |
| `val_small.json` | ~1 min | Eval |
| BigEarthNet Images | ~20 min | Generation eval only |

---

## STEP 1 — Clone the repo (first time only)

```bash
cd /teamspace/studios/this_studio
git clone https://github.com/tarunpanthi2006/SIH.git
cd SIH
git checkout feature/vlm
```

> **If you already have the repo, just pull latest:**
> ```bash
> cd /teamspace/studios/this_studio/SIH
> git pull origin feature/vlm
> ```

---

## STEP 2 — Install dependencies

```bash
pip install -r requirements.txt
pip install "accelerate==0.27.2"   # MUST pin this for transformers 4.36.x
```

---

## STEP 3 — Download LoRA Checkpoint from HuggingFace

This downloads your trained `checkpoint-1250` (the 0.6-loss model).

```bash
mkdir -p models/checkpoints/satquery-rs-vlm

python - <<'EOF'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Sh1vam26/satquery-rs-vlm",
    local_dir="models/checkpoints/satquery-rs-vlm",
    repo_type="model",
    token="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",
)
print("✅ Done!")
EOF
```

> After download, verify:
> ```bash
> ls models/checkpoints/satquery-rs-vlm/
> # You should see: checkpoint-1250/  (and maybe checkpoint-750/, checkpoint-1000/)
> ls models/checkpoints/satquery-rs-vlm/checkpoint-1250/
> # You should see: adapter_config.json  adapter_model.bin (or adapter_model.safetensors)
> ```

**Fix nested folder** (only needed if the ls above shows a `checkpoints/` subfolder):
```bash
mv models/checkpoints/satquery-rs-vlm/checkpoints/checkpoint-* \
   models/checkpoints/satquery-rs-vlm/
rmdir models/checkpoints/satquery-rs-vlm/checkpoints
```

---

## STEP 4 — Set environment variables

```bash
export HF_TOKEN="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"
export HF_HUB_REPO="Sh1vam26/satquery-rs-vlm"
export GEOCHAT_BASE_MODEL="MBZUAI/geochat-7B"
export LORA_ADAPTER_PATH="models/checkpoints/satquery-rs-vlm/checkpoint-1250"
export VISION_TOWER="openai/clip-vit-large-patch14-336"
export SATQUERY_MODE="quantized-4bit"
export DEVICE_MAP="auto"
```

> Or just copy your `.env` file to the Lightning AI machine.

---

## STEP 5A — Run Perplexity Evaluation (❌ NO images needed)

This compares base GeoChat-7B vs your adapted SatQuery-RS using **text-only loss**.  
No images, no data zip — just the JSON file which is already in the repo!

```bash
# val_small.json is already in the repo under datasets/bigearthnet/processed/
python -m training.finetuning.evaluate \
    --perplexity \
    --test-data datasets/bigearthnet/processed/val_small.json \
    --max-samples 100
```

**Expected output:**
```
adapted: avg_loss=0.6xxx, perplexity=~1.8
base:    avg_loss=1.8xxx, perplexity=~6.1
improvement: loss_reduction=1.2xxx (67% better!)
```

---

## STEP 5B — Run Full Generation Evaluation (✅ Needs images)

Only do this if you also want VQA accuracy and caption quality scores.

```bash
# First download the images (~20 min for val subset)
python training/bigearthnet/fast_image_download.py --subset val

# Then run the full comparison
python -m training.finetuning.evaluate \
    --compare \
    --test-data datasets/bigearthnet/processed/val_small.json \
    --max-samples 50
```

---

## STEP 6 (Optional) — Resume training if results not satisfactory

```bash
python -m training.finetuning.train \
    --config training/finetuning/config.yaml \
    --resume-from-hub
```

The new advanced training script will:
- Auto-download + patch the checkpoint for compatibility
- Show **live eval loss** every 250 steps alongside train loss
- Save `training_curves.csv` so you can plot the loss curve
- Auto-track the best checkpoint based on eval loss

---

## STEP 7 — Launch the API server

```bash
SATQUERY_MODE=quantized-4bit python -m backend.serve \
    --host 0.0.0.0 \
    --port 8080
```

Once you see `Application startup complete`, look for the **Port 8080** button in your Lightning AI UI — click it to get the public URL to hand to Person 1!

---

## ⚡ One-Shot Script

Instead of running the above steps manually, you can run the all-in-one setup script:

```bash
bash scripts/setup_lightning.sh
```

---

## 🔍 Troubleshooting

| Error | Fix |
|---|---|
| `best_global_step` KeyError | Run `git pull` — the new train.py auto-patches this |
| `weights_only` error on rng_state | Run `git pull` — the new train.py removes rng_state.pth automatically |
| `optimizer.pt` mismatch | Run `git pull` — the new train.py removes optimizer.pt automatically |
| Training running at 12s/it (slow) | Restart the Studio to flush RAM and reset GPU |
| `pixel_values` NameError in eval | Run `git pull` — evaluate.py now uses text-only perplexity |
