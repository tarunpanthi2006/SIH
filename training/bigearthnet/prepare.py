"""
BigEarthNet.txt — Download & Preparation Script
=================================================
Downloads:
1. BigEarthNet.txt parquet (text annotations) from HuggingFace
2. BigEarthNet v2.0 Sentinel-2 imagery from Zenodo (subset or full)

Usage:
    python -m training.bigearthnet.prepare --subset 100000
    python -m training.bigearthnet.prepare --full
"""

from __future__ import annotations

import argparse
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATASET_ROOT = Path("datasets/bigearthnet")
HF_DATASET_ID = "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt"
BEN_V2_ZENODO_S2 = "https://zenodo.org/records/10891137"


def download_parquet(output_dir: Path) -> Path:
    """
    Download the BigEarthNet.txt parquet file from HuggingFace.
    Returns path to the downloaded parquet file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("Install huggingface_hub: pip install huggingface-hub")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading BigEarthNet.txt parquet from HuggingFace: {HF_DATASET_ID}")

    try:
        parquet_path = hf_hub_download(
            repo_id=HF_DATASET_ID,
            filename="BigEarthNet.txt.parquet",
            repo_type="dataset",
            local_dir=str(output_dir / "raw"),
            local_dir_use_symlinks=False,
        )
        logger.info(f"Parquet downloaded to: {parquet_path}")
        return Path(parquet_path)
    except Exception as e:
        logger.error(f"Failed to download parquet: {e}")
        logger.info("You may need to: pip install huggingface-hub && huggingface-cli login")
        sys.exit(1)


def download_imagery_subset(
    output_dir: Path,
    parquet_path: Path,
    n_samples: int = 10000,
) -> Path:
    """
    Download a subset of BigEarthNet v2.0 Sentinel-2 imagery.

    For the full dataset (~70 GB), use the Zenodo download link directly.
    For a subset, we download individual patches referenced in our parquet subset.
    """
    import pandas as pd

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading parquet to identify {n_samples} image patches...")
    df = pd.read_parquet(parquet_path)

    # Get unique patch IDs (Sentinel-2)
    unique_patches = df["patch_id"].unique()
    logger.info(f"Total unique S2 patches in dataset: {len(unique_patches)}")

    if n_samples < len(unique_patches):
        # Sample a subset of patches
        import numpy as np
        rng = np.random.RandomState(42)
        selected_patches = rng.choice(unique_patches, size=min(n_samples, len(unique_patches)), replace=False)
    else:
        selected_patches = unique_patches

    logger.info(f"Selected {len(selected_patches)} patches for download")

    # Save the selected patch list for later use
    patch_list_path = output_dir / "selected_patches.txt"
    with open(patch_list_path, "w") as f:
        for patch_id in selected_patches:
            f.write(f"{patch_id}\n")

    logger.info(f"Patch list saved to: {patch_list_path}")
    logger.info(
        f"\n{'='*60}\n"
        f"MANUAL STEP REQUIRED:\n"
        f"Download BigEarthNet v2.0 Sentinel-2 imagery from:\n"
        f"  {BEN_V2_ZENODO_S2}\n"
        f"\n"
        f"Extract patches into:\n"
        f"  {images_dir.resolve()}\n"
        f"\n"
        f"Expected structure:\n"
        f"  {images_dir}/S2A_MSIL2A_..._patch/\n"
        f"    ├── S2A_..._B02.tif  (Blue)\n"
        f"    ├── S2A_..._B03.tif  (Green)\n"
        f"    ├── S2A_..._B04.tif  (Red)\n"
        f"    └── ...\n"
        f"{'='*60}"
    )

    return images_dir


def create_rgb_composites(
    images_dir: Path,
    output_dir: Path,
    max_patches: int | None = None,
) -> int:
    """
    Create RGB composite PNG images from Sentinel-2 bands.
    Uses B04 (Red), B03 (Green), B02 (Blue).

    These RGB composites are what the VLM's CLIP vision encoder expects.
    """
    try:
        import numpy as np
        from PIL import Image
        import rasterio
    except ImportError:
        try:
            import numpy as np
            from PIL import Image
            logger.warning(
                "rasterio not installed. Attempting to load .tif with PIL. "
                "For best results: pip install rasterio"
            )
            USE_RASTERIO = False
        except ImportError:
            logger.error("Install numpy and Pillow: pip install numpy Pillow")
            sys.exit(1)
    else:
        USE_RASTERIO = True

    composites_dir = output_dir / "rgb"
    composites_dir.mkdir(parents=True, exist_ok=True)

    # BigEarthNet v2.0 nests patches inside scene directories (e.g. images/scene_id/patch_id)
    # We use rglob to find all directories that actually contain a B04 file.
    patch_dirs = set(f.parent for f in images_dir.rglob("*_B04.tif"))
    patch_dirs = sorted(list(patch_dirs))
    
    if max_patches:
        patch_dirs = patch_dirs[:max_patches]

    created = 0
    for patch_dir in patch_dirs:
        patch_id = patch_dir.name
        output_path = composites_dir / f"{patch_id}.png"

        if output_path.exists():
            created += 1
            continue

        # Find B04, B03, B02 bands
        bands = {}
        for band_name in ["B04", "B03", "B02"]:
            band_files = list(patch_dir.glob(f"*_{band_name}.*"))
            if band_files:
                bands[band_name] = band_files[0]

        if len(bands) < 3:
            logger.warning(f"Skipping {patch_id}: missing RGB bands (found {list(bands.keys())})")
            continue

        try:
            if USE_RASTERIO:
                import rasterio
                band_arrays = []
                for band_name in ["B04", "B03", "B02"]:
                    with rasterio.open(bands[band_name]) as src:
                        band_arrays.append(src.read(1))
            else:
                band_arrays = []
                for band_name in ["B04", "B03", "B02"]:
                    img = Image.open(bands[band_name])
                    band_arrays.append(np.array(img))

            # Stack to RGB
            rgb = np.stack(band_arrays, axis=-1).astype(np.float32)

            # Normalize to 0-255 (clip at 2nd and 98th percentile for contrast)
            for c in range(3):
                p2 = np.percentile(rgb[:, :, c], 2)
                p98 = np.percentile(rgb[:, :, c], 98)
                if p98 > p2:
                    rgb[:, :, c] = np.clip((rgb[:, :, c] - p2) / (p98 - p2) * 255, 0, 255)
                else:
                    rgb[:, :, c] = 0

            rgb_img = Image.fromarray(rgb.astype(np.uint8))
            rgb_img.save(output_path)
            created += 1

        except Exception as e:
            logger.warning(f"Failed to create composite for {patch_id}: {e}")

    logger.info(f"Created {created} RGB composites in {composites_dir}")
    return created


def inspect_parquet(parquet_path: Path) -> None:
    """Print detailed statistics about the BigEarthNet.txt parquet."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)

    print(f"\n{'='*60}")
    print(f"BigEarthNet.txt Parquet Inspection")
    print(f"{'='*60}")
    print(f"Total rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"\n--- Column dtypes ---")
    print(df.dtypes)
    print(f"\n--- Split distribution ---")
    print(df["split"].value_counts())
    print(f"\n--- Task type distribution ---")
    print(df["type"].value_counts())
    print(f"\n--- Category distribution (top 20) ---")
    print(df["category"].value_counts().head(20))
    print(f"\n--- Unique patches ---")
    print(f"  Sentinel-2 patches: {df['patch_id'].nunique():,}")
    if "s1_name" in df.columns:
        print(f"  Sentinel-1 patches: {df['s1_name'].nunique():,}")
    print(f"\n--- Sample rows ---")
    print(df.head(5).to_string())
    print(f"\n--- Input length stats ---")
    print(df["input"].str.len().describe())
    print(f"\n--- Output length stats ---")
    print(df["output"].str.len().describe())
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare BigEarthNet.txt")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Output directory for dataset files",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Number of image patches to download (default: download parquet only)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Prepare for full dataset download",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only inspect an existing parquet file",
    )
    parser.add_argument(
        "--create-rgb",
        action="store_true",
        help="Create RGB composites from downloaded Sentinel-2 bands",
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=None,
        help="Path to existing parquet file (skip download)",
    )

    args = parser.parse_args()

    if args.parquet_path and args.parquet_path.exists():
        parquet_path = args.parquet_path
    elif args.inspect_only:
        default_path = args.output_dir / "raw" / "BigEarthNet.txt.parquet"
        if not default_path.exists():
            logger.error(f"Parquet not found at {default_path}. Download first.")
            sys.exit(1)
        parquet_path = default_path
    else:
        parquet_path = download_parquet(args.output_dir)

    # Always inspect
    inspect_parquet(parquet_path)

    if args.subset:
        download_imagery_subset(args.output_dir, parquet_path, n_samples=args.subset)
    elif args.full:
        download_imagery_subset(args.output_dir, parquet_path, n_samples=999_999_999)

    if args.create_rgb:
        images_dir = args.output_dir / "images"
        if images_dir.exists():
            create_rgb_composites(images_dir, args.output_dir)
        else:
            logger.warning(f"Images directory not found: {images_dir}. Download imagery first.")


if __name__ == "__main__":
    main()
