"""
BigEarthNet.txt — Parquet → LLaVA Instruction JSON Converter
==============================================================
Converts BigEarthNet.txt parquet annotations into the LLaVA-style
instruction-tuning JSON format required by GeoChat's LoRA fine-tuning.

Task types handled:
  - binary   → Yes/No VQA
  - mcq      → Multiple-choice VQA
  - captioning → Scene description
  - bounding box → Spatial grounding

Usage:
    python -m training.bigearthnet.convert \
        --parquet datasets/bigearthnet/raw/BigEarthNet.txt.parquet \
        --images-dir datasets/bigearthnet/rgb \
        --output-dir datasets/bigearthnet/processed \
        --max-samples 100000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================
# Conversion functions: parquet row → LLaVA conversation format
# =============================================================

def convert_binary_vqa(row: pd.Series, image_path: str) -> dict | None:
    """Convert a binary (yes/no) VQA row to LLaVA format."""
    question = row["input"].strip()
    answer = row["output"].strip()

    if not question or not answer:
        return None

    return {
        "id": f"ben_binary_{row.name}",
        "image": image_path,
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
    }


def convert_mcq_vqa(row: pd.Series, image_path: str) -> dict | None:
    """Convert a multiple-choice VQA row to LLaVA format."""
    question = row["input"].strip()
    answer = row["output"].strip()

    if not question or not answer:
        return None

    return {
        "id": f"ben_mcq_{row.name}",
        "image": image_path,
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
    }


def convert_captioning(row: pd.Series, image_path: str) -> dict | None:
    """Convert a captioning row to LLaVA format."""
    instruction = row["input"].strip()
    caption = row["output"].strip()

    if not caption:
        return None

    # Use a standard captioning prompt if the instruction is empty
    if not instruction:
        instruction = "Describe this satellite image in detail."

    return {
        "id": f"ben_cap_{row.name}",
        "image": image_path,
        "conversations": [
            {"from": "human", "value": f"<image>\n{instruction}"},
            {"from": "gpt", "value": caption},
        ],
    }


def convert_grounding(row: pd.Series, image_path: str) -> dict | None:
    """Convert a bounding box / grounding row to LLaVA format."""
    instruction = row["input"].strip()
    answer = row["output"].strip()

    if not instruction or not answer:
        return None

    return {
        "id": f"ben_ground_{row.name}",
        "image": image_path,
        "conversations": [
            {"from": "human", "value": f"<image>\n{instruction}"},
            {"from": "gpt", "value": answer},
        ],
    }


# Dispatcher: task type → conversion function
CONVERTERS = {
    "binary": convert_binary_vqa,
    "mcq": convert_mcq_vqa,
    "captioning": convert_captioning,
    "bounding box": convert_grounding,
}


def get_image_path(patch_id: str, images_dir: Path) -> str | None:
    """
    Resolve the image path for a given patch_id.
    Checks for RGB composite first, then raw patch directory.
    """
    # Check for pre-built RGB composite
    rgb_path = images_dir / f"{patch_id}.png"
    if rgb_path.exists():
        return str(rgb_path)

    rgb_path_jpg = images_dir / f"{patch_id}.jpg"
    if rgb_path_jpg.exists():
        return str(rgb_path_jpg)

    # Check for raw patch directory (user might not have created composites)
    patch_dir = images_dir.parent / "images" / patch_id
    if patch_dir.is_dir():
        return str(patch_dir)

    return None


def convert_dataset(
    parquet_path: Path,
    images_dir: Path,
    output_dir: Path,
    max_samples: int | None = None,
    require_images: bool = False,
    task_types: list[str] | None = None,
) -> dict[str, int]:
    """
    Convert BigEarthNet.txt parquet to LLaVA instruction JSON.

    Args:
        parquet_path: Path to BigEarthNet.txt.parquet
        images_dir: Path to RGB composite images directory
        output_dir: Where to save output JSON files
        max_samples: Maximum number of samples to convert (None = all)
        require_images: If True, skip rows where image is not found
        task_types: Filter to specific task types (None = all)

    Returns:
        Dictionary of statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df):,} rows")

    # Filter by task type if specified
    if task_types:
        df = df[df["type"].isin(task_types)]
        logger.info(f"Filtered to task types {task_types}: {len(df):,} rows")

    # Limit samples
    if max_samples and max_samples < len(df):
        # Stratified sampling to maintain task-type balance
        df = df.groupby("type", group_keys=False).apply(
            lambda x: x.sample(
                n=min(len(x), int(max_samples * len(x) / len(df))),
                random_state=42,
            )
        ).reset_index(drop=True)
        logger.info(f"Sampled {len(df):,} rows (stratified by task type)")

    # Convert rows
    all_samples: list[dict] = []
    stats = {"total": 0, "converted": 0, "skipped_no_image": 0, "skipped_conversion": 0}
    type_counts: dict[str, int] = {}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
        stats["total"] += 1

        task_type = row.get("type", "unknown")
        patch_id = row.get("patch_id", "")

        # Resolve image path
        image_path = get_image_path(patch_id, images_dir)
        if require_images and image_path is None:
            stats["skipped_no_image"] += 1
            continue

        # Use a placeholder path if images not downloaded yet
        if image_path is None:
            image_path = f"datasets/bigearthnet/rgb/{patch_id}.png"

        # Convert using the appropriate converter
        converter = CONVERTERS.get(task_type)
        if converter is None:
            # Default: treat as generic VQA
            converter = convert_binary_vqa

        sample = converter(row, image_path)
        if sample is None:
            stats["skipped_conversion"] += 1
            continue

        # Add metadata for provenance tracking
        sample["metadata"] = {
            "source": "bigearthnet_txt",
            "task_type": task_type,
            "category": row.get("category", ""),
            "patch_id": patch_id,
            "split": row.get("split", ""),
        }

        all_samples.append(sample)
        stats["converted"] += 1
        type_counts[task_type] = type_counts.get(task_type, 0) + 1

    # Save as single JSON file
    output_path = output_dir / "bigearthnet_instructions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    logger.info(f"\nConversion complete!")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Total processed: {stats['total']:,}")
    logger.info(f"  Converted: {stats['converted']:,}")
    logger.info(f"  Skipped (no image): {stats['skipped_no_image']:,}")
    logger.info(f"  Skipped (conversion): {stats['skipped_conversion']:,}")
    logger.info(f"  By task type: {json.dumps(type_counts, indent=4)}")

    # Save stats
    stats["type_counts"] = type_counts
    stats_path = output_dir / "conversion_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert BigEarthNet.txt to LLaVA format")
    parser.add_argument(
        "--parquet",
        type=Path,
        required=True,
        help="Path to BigEarthNet.txt.parquet",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("datasets/bigearthnet/rgb"),
        help="Path to RGB composite images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/bigearthnet/processed"),
        help="Output directory for instruction JSON",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to convert",
    )
    parser.add_argument(
        "--require-images",
        action="store_true",
        help="Skip rows where image file is not found",
    )
    parser.add_argument(
        "--task-types",
        nargs="+",
        default=None,
        choices=["binary", "mcq", "captioning", "bounding box"],
        help="Filter to specific task types",
    )

    args = parser.parse_args()
    convert_dataset(
        parquet_path=args.parquet,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        require_images=args.require_images,
        task_types=args.task_types,
    )


if __name__ == "__main__":
    main()
