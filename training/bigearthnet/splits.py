"""
BigEarthNet.txt — Train/Val/Test Split Generator
==================================================
Creates splits from the converted instruction JSON,
respecting the original BigEarthNet.txt split assignments.

Also generates small debug subsets for fast iteration.

Usage:
    python -m training.bigearthnet.splits \
        --data datasets/bigearthnet/processed/bigearthnet_instructions.json \
        --output-dir datasets/bigearthnet/processed
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def create_splits(
    data_path: Path,
    output_dir: Path,
    debug_size: int = 1000,
    small_size: int = 10000,
) -> dict[str, int]:
    """
    Create train/val/test splits from instruction JSON.

    Uses the 'split' field in each sample's metadata (from the
    original BigEarthNet.txt parquet) to assign splits.

    Also creates:
    - debug split: tiny subset for fast iteration
    - small split: medium subset for initial LoRA training
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    logger.info(f"Total samples: {len(all_samples):,}")

    # Split by the metadata.split field
    splits: dict[str, list[dict]] = {
        "train": [],
        "validation": [],
        "test": [],
        "bench": [],
        "unknown": [],
    }

    for sample in all_samples:
        split_name = sample.get("metadata", {}).get("split", "unknown")
        if split_name in splits:
            splits[split_name].append(sample)
        else:
            splits["unknown"].append(sample)

    # If no split info available, create 80/10/10 split
    if len(splits["train"]) == 0 and len(splits["unknown"]) > 0:
        logger.info("No split metadata found — creating 80/10/10 split")
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(all_samples))

        n = len(all_samples)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)

        splits["train"] = [all_samples[i] for i in indices[:train_end]]
        splits["validation"] = [all_samples[i] for i in indices[train_end:val_end]]
        splits["test"] = [all_samples[i] for i in indices[val_end:]]
        splits["unknown"] = []

    # Merge 'bench' into 'test' if bench exists
    if splits["bench"]:
        logger.info(f"Merging {len(splits['bench'])} bench samples into test set")
        splits["test"].extend(splits["bench"])

    # Save main splits
    stats = {}
    for split_name in ["train", "validation", "test"]:
        split_data = splits[split_name]
        if not split_data:
            continue

        split_path = output_dir / f"{split_name}.json"
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)

        stats[split_name] = len(split_data)
        logger.info(f"  {split_name}: {len(split_data):,} samples → {split_path}")

    # Create debug subset (tiny, for fast iteration)
    rng = np.random.RandomState(42)
    train_data = splits["train"]

    if train_data and debug_size > 0:
        debug_indices = rng.choice(
            len(train_data),
            size=min(debug_size, len(train_data)),
            replace=False,
        )
        debug_data = [train_data[i] for i in debug_indices]
        debug_path = output_dir / "debug.json"
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug_data, f, indent=2, ensure_ascii=False)
        stats["debug"] = len(debug_data)
        logger.info(f"  debug: {len(debug_data):,} samples → {debug_path}")

    # Create small subset (for initial LoRA training)
    if train_data and small_size > 0:
        small_indices = rng.choice(
            len(train_data),
            size=min(small_size, len(train_data)),
            replace=False,
        )
        small_data = [train_data[i] for i in small_indices]
        small_path = output_dir / "train_small.json"
        with open(small_path, "w", encoding="utf-8") as f:
            json.dump(small_data, f, indent=2, ensure_ascii=False)
        stats["train_small"] = len(small_data)
        logger.info(f"  train_small: {len(small_data):,} samples → {small_path}")

    # Also save a val subset for fast evaluation during training
    val_data = splits["validation"]
    if val_data:
        val_small_size = min(500, len(val_data))
        val_small_indices = rng.choice(len(val_data), size=val_small_size, replace=False)
        val_small_data = [val_data[i] for i in val_small_indices]
        val_small_path = output_dir / "val_small.json"
        with open(val_small_path, "w", encoding="utf-8") as f:
            json.dump(val_small_data, f, indent=2, ensure_ascii=False)
        stats["val_small"] = len(val_small_data)
        logger.info(f"  val_small: {len(val_small_data):,} samples → {val_small_path}")

    # Print task-type distribution per split
    logger.info(f"\n--- Task distribution per split ---")
    for split_name in ["train", "validation", "test"]:
        if not splits[split_name]:
            continue
        type_counts = Counter(
            s.get("metadata", {}).get("task_type", "unknown")
            for s in splits[split_name]
        )
        logger.info(f"  {split_name}:")
        for task_type, count in type_counts.most_common():
            logger.info(f"    {task_type}: {count:,}")

    # Save split statistics
    stats_path = output_dir / "split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"\nSplit stats saved to: {stats_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to converted instruction JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/bigearthnet/processed"),
        help="Output directory for split files",
    )
    parser.add_argument(
        "--debug-size",
        type=int,
        default=1000,
        help="Size of debug subset",
    )
    parser.add_argument(
        "--small-size",
        type=int,
        default=10000,
        help="Size of small training subset",
    )

    args = parser.parse_args()
    create_splits(
        data_path=args.data,
        output_dir=args.output_dir,
        debug_size=args.debug_size,
        small_size=args.small_size,
    )


if __name__ == "__main__":
    main()
