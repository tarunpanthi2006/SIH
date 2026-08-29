"""
BigEarthNet.txt — Data Validation Script
==========================================
Validates the converted instruction JSON for:
- Image-text alignment
- Missing/broken references
- Schema conformance
- Task distribution balance
- Answer quality checks

Usage:
    python -m training.bigearthnet.validate \
        --data datasets/bigearthnet/processed/bigearthnet_instructions.json \
        --images-dir datasets/bigearthnet/rgb \
        --check-images
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def validate_schema(sample: dict, idx: int) -> list[str]:
    """Validate a single instruction sample against the expected schema."""
    errors = []

    required_fields = ["id", "image", "conversations"]
    for field in required_fields:
        if field not in sample:
            errors.append(f"Sample {idx}: missing required field '{field}'")

    if "conversations" in sample:
        convs = sample["conversations"]
        if not isinstance(convs, list) or len(convs) < 2:
            errors.append(f"Sample {idx}: conversations must have ≥2 turns")
        else:
            if convs[0].get("from") != "human":
                errors.append(f"Sample {idx}: first turn must be from 'human'")
            if convs[1].get("from") != "gpt":
                errors.append(f"Sample {idx}: second turn must be from 'gpt'")
            if not convs[0].get("value", "").strip():
                errors.append(f"Sample {idx}: human turn is empty")
            if not convs[1].get("value", "").strip():
                errors.append(f"Sample {idx}: gpt turn is empty")
            if "<image>" not in convs[0].get("value", ""):
                errors.append(f"Sample {idx}: human turn missing '<image>' token")

    return errors


def validate_image_exists(sample: dict, idx: int, images_dir: Path) -> list[str]:
    """Check if the referenced image file exists on disk."""
    errors = []
    image_path = sample.get("image", "")

    if not image_path:
        errors.append(f"Sample {idx}: empty image path")
        return errors

    # Check absolute and relative paths
    full_path = Path(image_path)
    if not full_path.exists():
        rel_path = images_dir / Path(image_path).name
        if not rel_path.exists():
            errors.append(f"Sample {idx}: image not found: {image_path}")

    return errors


def compute_statistics(samples: list[dict]) -> dict:
    """Compute comprehensive statistics about the dataset."""
    stats = {
        "total_samples": len(samples),
        "unique_images": len(set(s.get("image", "") for s in samples)),
    }

    # Task type distribution
    task_types = Counter()
    for s in samples:
        meta = s.get("metadata", {})
        task_types[meta.get("task_type", "unknown")] += 1
    stats["task_distribution"] = dict(task_types)

    # Split distribution
    splits = Counter()
    for s in samples:
        meta = s.get("metadata", {})
        splits[meta.get("split", "unknown")] += 1
    stats["split_distribution"] = dict(splits)

    # Answer length statistics
    answer_lengths = []
    question_lengths = []
    for s in samples:
        convs = s.get("conversations", [])
        if len(convs) >= 2:
            q_len = len(convs[0].get("value", ""))
            a_len = len(convs[1].get("value", ""))
            question_lengths.append(q_len)
            answer_lengths.append(a_len)

    if answer_lengths:
        stats["answer_length"] = {
            "min": min(answer_lengths),
            "max": max(answer_lengths),
            "mean": round(sum(answer_lengths) / len(answer_lengths), 1),
            "median": sorted(answer_lengths)[len(answer_lengths) // 2],
        }
        stats["question_length"] = {
            "min": min(question_lengths),
            "max": max(question_lengths),
            "mean": round(sum(question_lengths) / len(question_lengths), 1),
            "median": sorted(question_lengths)[len(question_lengths) // 2],
        }

    # Category distribution (top 20)
    categories = Counter()
    for s in samples:
        meta = s.get("metadata", {})
        categories[meta.get("category", "unknown")] += 1
    stats["top_categories"] = dict(categories.most_common(20))

    return stats


def validate_dataset(
    data_path: Path,
    images_dir: Path | None = None,
    check_images: bool = False,
    max_errors: int = 50,
) -> bool:
    """
    Run full validation on the instruction JSON dataset.
    Returns True if validation passes, False otherwise.
    """
    logger.info(f"Validating dataset: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    logger.info(f"Loaded {len(samples):,} samples")

    # Schema validation
    all_errors = []
    for idx, sample in enumerate(samples):
        errors = validate_schema(sample, idx)
        all_errors.extend(errors)

        if check_images and images_dir:
            errors = validate_image_exists(sample, idx, images_dir)
            all_errors.extend(errors)

        if len(all_errors) >= max_errors:
            logger.warning(f"Stopping after {max_errors} errors")
            break

    # Print errors
    if all_errors:
        logger.warning(f"\n{'='*60}")
        logger.warning(f"VALIDATION ERRORS: {len(all_errors)}")
        logger.warning(f"{'='*60}")
        for error in all_errors[:max_errors]:
            logger.warning(f"  ❌ {error}")
        if len(all_errors) > max_errors:
            logger.warning(f"  ... and {len(all_errors) - max_errors} more errors")
    else:
        logger.info(f"✅ Schema validation passed — no errors found")

    # Compute and print statistics
    stats = compute_statistics(samples)
    logger.info(f"\n{'='*60}")
    logger.info(f"DATASET STATISTICS")
    logger.info(f"{'='*60}")
    logger.info(f"  Total samples: {stats['total_samples']:,}")
    logger.info(f"  Unique images: {stats['unique_images']:,}")
    logger.info(f"\n  Task distribution:")
    for task, count in stats.get("task_distribution", {}).items():
        pct = count / stats["total_samples"] * 100
        logger.info(f"    {task:20s}: {count:>8,} ({pct:.1f}%)")
    logger.info(f"\n  Split distribution:")
    for split, count in stats.get("split_distribution", {}).items():
        pct = count / stats["total_samples"] * 100
        logger.info(f"    {split:20s}: {count:>8,} ({pct:.1f}%)")
    if "answer_length" in stats:
        al = stats["answer_length"]
        logger.info(f"\n  Answer length: min={al['min']}, max={al['max']}, mean={al['mean']}, median={al['median']}")
    logger.info(f"{'='*60}\n")

    # Save validation report
    report_path = data_path.parent / "validation_report.json"
    report = {"errors": all_errors[:100], "statistics": stats, "passed": len(all_errors) == 0}
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Validation report saved to: {report_path}")

    return len(all_errors) == 0


def main():
    parser = argparse.ArgumentParser(description="Validate BigEarthNet.txt instruction JSON")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to instruction JSON file",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Path to images directory for image existence checks",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Verify that referenced images exist on disk",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=50,
        help="Maximum errors to report before stopping",
    )

    args = parser.parse_args()

    passed = validate_dataset(
        data_path=args.data,
        images_dir=args.images_dir,
        check_images=args.check_images,
        max_errors=args.max_errors,
    )

    if not passed:
        logger.error("Validation FAILED")
        exit(1)
    else:
        logger.info("Validation PASSED ✅")


if __name__ == "__main__":
    main()
