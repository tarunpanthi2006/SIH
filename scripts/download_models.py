"""
SatQuery — Model / Checkpoint Downloader
==========================================

Fetches all specialist-layer checkpoints into checkpoints/:

    checkpoints/changeformer/ChangeFormer_LEVIR.pth
    checkpoints/skysensepp/...
    checkpoints/prithvi/...

Every download is followed by a sanity check (file exists, is non-trivially
sized, and — where possible — is actually loadable by torch/safetensors)
so a broken or placeholder file fails LOUDLY here instead of surfacing
three files downstream as a cryptic "Failed to load model" error.

IMPORTANT — ChangeFormer:
--------------------------
The official wgcban/ChangeFormer LEVIR-CD checkpoint ("ChangeFormerV6") is
NOT on Google Drive and NOT on HuggingFace — it's a GitHub Releases .zip:

    https://github.com/wgcban/ChangeFormer/releases/download/v0.1.0/
    CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_
    multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256.zip

(confirmed directly from the repo README as of the current release; if
wgcban ever cuts a new release this URL may change — re-check the
"Github-LEVIR-Pretrained" link under the LEVIR-CD quick-start section if
this 404s). It unzips to a folder containing a `best_ckpt.pt` (per the
repo's own eval script conventions) — this script downloads the zip,
extracts it, and locates that file automatically.

Do NOT use the Dropbox link
(https://www.dropbox.com/s/undtrlxiz7bkag5/pretrained_changeformer.pt) —
that is only the ImageNet/ADE-pretrained SegFormer backbone used to
*initialize* training, not the final trained change-detection weights.

This script will refuse to write a placeholder file — if the real zip
can't be fetched, ChangeFormer download fails loudly instead of silently
producing a fake checkpoint (that was the bug in the previous version of
this script: it wrote 14 bytes of literal garbage and called it done).

IMPORTANT — SkySense++ CANNOT be auto-downloaded:
---------------------------------------------------
`kang-wu/SkySensePlusPlus` is a GITHUB code repo, not a HuggingFace model
repo — there is no `huggingface.co/kang-wu/SkySensePlusPlus`, so any
`snapshot_download(repo_id="kang-wu/SkySensePlusPlus")` call 404s
(confirmed directly against the repo). Per that repo's own README, every
checkpoint link (pretraining weights AND all downstream/finetuned
weights) points to a single Notion page:

    https://www.notion.so/SkySense-Checkpoints-a7fcff6ce29a4647a08c7fe416910509

That's a manual, click-through webpage — there is no stable, scriptable
file URL behind it that this downloader can hit. This script therefore
does NOT attempt to download SkySense++ automatically. It only verifies
whether a human has already placed a real checkpoint file under
checkpoints/skysensepp/, and if not, fails with instructions rather than
silently trying (and failing) an HF API call.

Also note: the official SkySense++ code is built on Alibaba's `antmmf`
framework + mmcv-full==1.7.1 + mmsegmentation==0.30.0 + torch==1.13.1 +
GDAL — NOT the simplified dual-ViT stand-in architecture used in
`models/optical_sar/model.py` here. Even with the real checkpoint in
hand, `SkySensePPModel.from_pretrained`'s key-overlap check will very
likely refuse it (by design — see that file's docstring) unless someone
writes real parameter-name remapping or ports the official model code.
The pre-trained weights are licensed for non-commercial research only
(contact yansheng.li@whu.edu.cn at Wuhan University for commercial use).
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

# ── Config ───────────────────────────────────────────────────────────────

CKPT_DIR = Path("checkpoints")

# GitHub Releases .zip for the official LEVIR-CD ChangeFormerV6 checkpoint.
# Override via env var if wgcban publishes a new release with a different URL.
CHANGEFORMER_CHECKPOINT_URL = os.environ.get(
    "CHANGEFORMER_CHECKPOINT_URL",
    "https://github.com/wgcban/ChangeFormer/releases/download/v0.1.0/"
    "CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_"
    "multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256.zip",
)

# Where a human must go to get SkySense++ weights — no scriptable URL exists.
SKYSENSEPP_NOTION_URL = (
    "https://www.notion.so/SkySense-Checkpoints-a7fcff6ce29a4647a08c7fe416910509"
)
SKYSENSEPP_GITHUB_URL = "https://github.com/kang-wu/SkySensePlusPlus"

# Minimum plausible size (bytes) for each checkpoint type. Real weights for
# these models are all well into the tens-to-hundreds of MB; anything under
# a few MB is almost certainly a broken/partial/placeholder download.
MIN_SIZE_BYTES = {
    "changeformer": 5 * 1024 * 1024,      # ChangeFormerV6 (~50-100MB typical)
    "skysensepp": 50 * 1024 * 1024,       # foundation-model scale
    "prithvi": 500 * 1024 * 1024,         # Prithvi-EO-2.0-600M is large
}


# ── Sanity checks ────────────────────────────────────────────────────────

def _check_file_size(path: Path, min_bytes: int, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"[{label}] Expected file not found: {path}")
    size = path.stat().st_size
    if size < min_bytes:
        raise ValueError(
            f"[{label}] {path} is only {size:,} bytes "
            f"(expected at least {min_bytes:,}). This looks like a broken "
            f"or placeholder download, not real model weights. Refusing to "
            f"treat this as a valid checkpoint."
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
            f"under {dir_path}. Downloaded repo may only contain configs, "
            f"datasets, or documentation — inspect {dir_path} manually."
        )
    total_size = sum(f.stat().st_size for f in weight_files)
    if total_size < min_bytes:
        raise ValueError(
            f"[{label}] Weight files under {dir_path} total only "
            f"{total_size / 1e6:.1f} MB (expected at least "
            f"{min_bytes / 1e6:.0f} MB). Likely an incomplete download."
        )
    print(f"  [OK] {len(weight_files)} weight file(s), "
          f"{total_size / 1e6:.1f} MB total under {dir_path}")


def _try_load_with_torch(path: Path, label: str) -> None:
    """Best-effort: confirm the file is actually a loadable checkpoint,
    not just big-enough garbage. Only run this on single-file checkpoints
    (.pth/.pt) — safetensors/HF snapshots are checked structurally instead.
    """
    import torch
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        n_keys = len(state) if isinstance(state, dict) else "?"
        print(f"  [OK] {label}: torch.load succeeded ({n_keys} top-level keys)")
    except Exception as exc:
        raise ValueError(
            f"[{label}] {path} exists and is large enough, but torch.load "
            f"failed: {exc}. This is not a valid PyTorch checkpoint."
        ) from exc


# ── Downloaders ──────────────────────────────────────────────────────────

def _download_with_progress(url: str, dest: Path) -> None:
    """Plain streaming HTTPS download (GitHub Releases assets don't need
    gdown/Drive handling — they're direct-downloadable)."""
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
    """Search an extracted checkpoint folder for the actual weights file.
    The official zip nests things under a long descriptive folder name and
    the real file is typically `best_ckpt.pt` (per the repo's own eval
    scripts), but fall back to any .pt/.pth if that exact name isn't found.
    """
    exact = list(root.rglob("best_ckpt.pt"))
    if exact:
        return exact[0]
    candidates = list(root.rglob("*.pt")) + list(root.rglob("*.pth"))
    if not candidates:
        return None
    # Prefer the largest file — checkpoints dwarf any stray config/log files.
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
        raise FileNotFoundError(
            f"Downloaded and extracted {zip_path.name}, but found no "
            f".pt/.pth file inside {extract_dir}. Inspect the extracted "
            f"contents manually — the release's internal layout may have "
            f"changed."
        )

    shutil.copy2(ckpt_file, final_path)
    zip_path.unlink()
    shutil.rmtree(extract_dir)

    _check_file_size(final_path, MIN_SIZE_BYTES["changeformer"], "changeformer")
    _try_load_with_torch(final_path, "changeformer")


def download_skysensepp() -> None:
    """
    SkySense++ has NO scriptable download source (see module docstring):
    `kang-wu/SkySensePlusPlus` is a GitHub code repo, not an HF model repo,
    and every checkpoint link in that repo's README points to a manual
    Notion page. This function therefore does NOT attempt any network
    fetch — it only checks whether a human has already placed a real
    checkpoint file under checkpoints/skysensepp/, and fails with clear
    manual instructions if not.
    """
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
            No SkySense++ checkpoint found under {out_dir}, and this
            script CANNOT fetch it automatically — there is no HF model
            repo and no stable scriptable URL for it.

            To get it manually:
              1. Open {SKYSENSEPP_NOTION_URL}
                 (linked from the official repo: {SKYSENSEPP_GITHUB_URL})
              2. Download the checkpoint file you need (pretraining
                 weights, or a downstream/finetuned checkpoint).
              3. Place it under: {out_dir}/

            Note: these weights are licensed for non-commercial research
            only (contact yansheng.li@whu.edu.cn for commercial use), and
            the official architecture (antmmf + mmsegmentation, built on
            torch==1.13.1) does NOT match the simplified stand-in model
            in models/optical_sar/model.py — expect
            SkySensePPCheckpointMismatch when you try to load it unless
            you port the real model code or write key remapping.

            If you're fine running the specialist layer with the
            stand-in architecture's random-init weights for now (e.g.
            for a demo where SkySense++ isn't the focus), skip this
            checkpoint and don't call run_optical_sar() until it's
            resolved.
        """).strip())

    _check_dir_has_real_weights(out_dir, MIN_SIZE_BYTES["skysensepp"], "skysensepp")


def download_prithvi() -> None:
    print("Downloading Prithvi-EO-2.0-600M...")
    out_dir = CKPT_DIR / "prithvi"
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id="ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
        local_dir=str(out_dir),
    )
    _check_dir_has_real_weights(out_dir, MIN_SIZE_BYTES["prithvi"], "prithvi")


# ── Entry point ──────────────────────────────────────────────────────────

def download_models() -> None:
    CKPT_DIR.mkdir(exist_ok=True)
    failures: list[str] = []

    for name, fn in [
        ("changeformer", download_changeformer),
        ("skysensepp", download_skysensepp),
        ("prithvi", download_prithvi),
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