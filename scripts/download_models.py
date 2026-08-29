import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

def download_models():
    print("Downloading models...")
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    # ChangeFormer (mock or use specific HF mirror if available, for now assume huggingface repo)
    print("Downloading ChangeFormer...")
    # Using a placeholder since ChangeFormer weights are typically gdrive
    changeformer_dir = ckpt_dir / "changeformer"
    changeformer_dir.mkdir(exist_ok=True)
    with open(changeformer_dir / "ChangeFormer_LEVIR.pth", "wb") as f:
        # Mock file since actual is gdrive. For real use gdown.
        f.write(b"dummy_weights")
        
    print("Downloading SkySense++...")
    snapshot_download(repo_id="kang-wu/SkySensePlusPlus", local_dir=ckpt_dir / "skysensepp")

    print("Downloading Prithvi-EO-2.0-600M...")
    snapshot_download(repo_id="ibm-nasa-geospatial/Prithvi-EO-2.0-600M", local_dir=ckpt_dir / "prithvi")
    print("Done!")

if __name__ == "__main__":
    download_models()
