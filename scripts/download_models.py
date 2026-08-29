"""
<<<<<<< HEAD
SatQuery — Model / Checkpoint Downloader
==========================================

Fetches all specialist-layer checkpoints into checkpoints/:

    checkpoints/changeformer/ChangeFormer_LEVIR.pth
    checkpoints/skysensepp/...
    checkpoints/prithvi/...
    checkpoints/satquery-rs-vlm/...

Every download is followed by a sanity check.
"""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
import zipfile
from pathlib import Path

import requests
from huggingface_hub import snapshot_download
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────

CKPT_DIR = Path("checkpoints")
VLM_DIR = Path("models/checkpoints/satquery-rs-vlm")

CHANGEFORMER_CHECKPOINT_URL = os.environ.get(
    "CHANGEFORMER_CHECKPOINT_URL",
    "https://github.com/wgcban/ChangeFormer/releases/download/v0.1.0/"
    "CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_"
    "multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256.zip",
)

SKYSENSEPP_NOTION_URL = (
    "https://www.notion.so/SkySense-Checkpoints-a7fcff6ce29a4647a08c7fe416910509"
)
SKYSENSEPP_GITHUB_URL = "https://github.com/kang-wu/SkySensePlusPlus"

MIN_SIZE_BYTES = {
    "changeformer": 5 * 1024 * 1024,
    "skysensepp": 50 * 1024 * 1024,
    "prithvi": 500 * 1024 * 1024,
}

GEOCHAT_MODEL_ID = "MBZUAI/geochat-7B"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"


# ── Sanity checks ────────────────────────────────────────────────────────

def _check_file_size(path: Path, min_bytes: int, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"[{label}] Expected file not found: {path}")
    size = path.stat().st_size
    if size < min_bytes:
        raise ValueError(
            f"[{label}] {path} is only {size:,} bytes "
            f"(expected at least {min_bytes:,})."
        )
    print(f"  [OK] {path} ({size / 1e6:.1f} MB)")


def _check_dir_has_real_weights(dir_path: Path, min_bytes: int, label: str) -> None:
    if not dir_path.exists() or not any(dir_path.iterdir()):
        raise FileNotFoundError(
            f"[{label}] {dir_path} is missing or empty after download."
        )
    weight_files = (
        list(dir_path.rglob("*.pth"))
        + list(dir_path.rglob("*.pt"))
        + list(dir_path.rglob("*.safetensors"))
        + list(dir_path.rglob("*.bin"))
    )
    if not weight_files:
        raise FileNotFoundError(
            f"[{label}] No weight files (.pth/.pt/.safetensors/.bin) found "
            f"under {dir_path}."
        )
    total_size = sum(f.stat().st_size for f in weight_files)
    if total_size < min_bytes:
        raise ValueError(
            f"[{label}] Weight files under {dir_path} total only "
            f"{total_size / 1e6:.1f} MB."
        )
    print(f"  [OK] {len(weight_files)} weight file(s), "
          f"{total_size / 1e6:.1f} MB total under {dir_path}")


def _try_load_with_torch(path: Path, label: str) -> None:
    import torch
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        n_keys = len(state) if isinstance(state, dict) else "?"
        print(f"  [OK] {label}: torch.load succeeded ({n_keys} top-level keys)")
    except Exception as exc:
        raise ValueError(
            f"[{label}] {path} torch.load failed: {exc}."
        ) from exc


# ── Downloaders ──────────────────────────────────────────────────────────

def _download_with_progress(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        written = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                written += len(chunk)
                if total:
                    pct = written / total * 100
                    print(f"\r  {written / 1e6:.1f}/{total / 1e6:.1f} MB "
                          f"({pct:.0f}%)", end="", flush=True)
    print()


def _find_checkpoint_in_dir(root: Path) -> Path | None:
    exact = list(root.rglob("best_ckpt.pt"))
    if exact:
        return exact[0]
    candidates = list(root.rglob("*.pt")) + list(root.rglob("*.pth"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def download_changeformer() -> None:
    print("Downloading ChangeFormer (LEVIR-CD)...")
    out_dir = CKPT_DIR / "changeformer"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "ChangeFormer_LEVIR.pth"

    zip_path = out_dir / "_download.zip"
    extract_dir = out_dir / "_extracted"

    print(f"  Fetching {CHANGEFORMER_CHECKPOINT_URL}")
    _download_with_progress(CHANGEFORMER_CHECKPOINT_URL, zip_path)

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    ckpt_file = _find_checkpoint_in_dir(extract_dir)
    if ckpt_file is None:
        raise FileNotFoundError(f"No .pt file found inside {extract_dir}.")

    shutil.copy2(ckpt_file, final_path)
    zip_path.unlink()
    shutil.rmtree(extract_dir)

    _check_file_size(final_path, MIN_SIZE_BYTES["changeformer"], "changeformer")
    _try_load_with_torch(final_path, "changeformer")


def download_skysensepp() -> None:
    print("Checking SkySense++ (no automated download available)...")
    out_dir = CKPT_DIR / "skysensepp"
    out_dir.mkdir(parents=True, exist_ok=True)

    weight_files = (
        list(out_dir.rglob("*.pth"))
        + list(out_dir.rglob("*.pt"))
        + list(out_dir.rglob("*.safetensors"))
        + list(out_dir.rglob("*.bin"))
    )

    if not weight_files:
        raise FileNotFoundError(textwrap.dedent(f"""
            No SkySense++ checkpoint found under {out_dir}.
            Manual download required from: {SKYSENSEPP_NOTION_URL}
        """).strip())

    _check_dir_has_real_weights(out_dir, MIN_SIZE_BYTES["skysensepp"], "skysensepp")


def download_prithvi() -> None:
    print("Downloading Prithvi-EO-2.0-600M...")
    out_dir = CKPT_DIR / "prithvi"
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(repo_id="ibm-nasa-geospatial/Prithvi-EO-2.0-600M", local_dir=str(out_dir))
    _check_dir_has_real_weights(out_dir, MIN_SIZE_BYTES["prithvi"], "prithvi")


def download_vlm_base() -> None:
    print("Downloading GeoChat-7B base model...")
    token = os.getenv("HF_TOKEN", None)
    try:
        snapshot_download(repo_id=GEOCHAT_MODEL_ID, token=token, resume_download=True)
        print("  [OK] GeoChat-7B")
    except Exception as e:
        raise RuntimeError(f"Failed to download GeoChat-7B: {e}")


def download_vlm_vision() -> None:
    print("Downloading CLIP vision tower...")
    try:
        snapshot_download(repo_id=CLIP_MODEL_ID, resume_download=True)
        print("  [OK] CLIP")
    except Exception as e:
        raise RuntimeError(f"Failed to download CLIP: {e}")


def download_vlm_adapter() -> None:
    hub_repo = os.getenv("HF_HUB_REPO", "")
    if not hub_repo:
        print("  [SKIP] No HF_HUB_REPO specified for SatQuery-RS adapter.")
        return
    print(f"Downloading SatQuery-RS adapter from {hub_repo}...")
    VLM_DIR.mkdir(parents=True, exist_ok=True)
    token = os.getenv("HF_TOKEN", None)
    try:
        snapshot_download(repo_id=hub_repo, local_dir=str(VLM_DIR), token=token, resume_download=True)
        print("  [OK] SatQuery-RS Adapter")
    except Exception as e:
        raise RuntimeError(f"Failed to download adapter: {e}")


# ── Entry point ──────────────────────────────────────────────────────────

def download_models() -> None:
    CKPT_DIR.mkdir(exist_ok=True)
    failures: list[str] = []

    for name, fn in [
        ("changeformer", download_changeformer),
        ("skysensepp", download_skysensepp),
        ("prithvi", download_prithvi),
        ("vlm_base", download_vlm_base),
        ("vlm_vision", download_vlm_vision),
        ("vlm_adapter", download_vlm_adapter),
    ]:
        try:
            fn()
        except Exception as exc:
            print(f"  [FAILED] {name}: {exc}")
            failures.append(name)
        print()

    if failures:
        print(f"Done with failures: {', '.join(failures)}. "
              f"Fix these before running any specialist tool against them.")
        sys.exit(1)
    print("All checkpoints downloaded and verified.")


if __name__ == "__main__":
    download_models()
