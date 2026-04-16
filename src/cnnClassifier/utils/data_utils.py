import hashlib
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from cnnClassifier import logger

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_grouped_split_dataframe(
    dataset_dir: Path,
    validation_split: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    train_rows: list[dict] = []
    validation_rows: list[dict] = []
    duplicate_summary: dict[str, dict[str, int]] = {}

    for class_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        grouped_by_hash: dict[str, list[Path]] = defaultdict(list)
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            grouped_by_hash[_file_md5(image_path)].append(image_path)

        unique_groups = list(grouped_by_hash.values())
        if not unique_groups:
            continue

        rng = random.Random(seed)
        rng.shuffle(unique_groups)

        if len(unique_groups) == 1:
            validation_group_count = 1
        else:
            validation_group_count = max(1, round(len(unique_groups) * validation_split))
            validation_group_count = min(validation_group_count, len(unique_groups) - 1)

        validation_groups = unique_groups[:validation_group_count]
        training_groups = unique_groups[validation_group_count:]

        # Keep one representative per exact duplicate group to avoid leakage and overcounting.
        for group in training_groups:
            train_rows.append({"filepath": str(group[0]), "label": class_dir.name})

        for group in validation_groups:
            validation_rows.append({"filepath": str(group[0]), "label": class_dir.name})

        total_images = sum(len(group) for group in unique_groups)
        duplicate_summary[class_dir.name] = {
            "total_images": total_images,
            "unique_images": len(unique_groups),
            "duplicates_removed": total_images - len(unique_groups),
            "training_unique_images": len(training_groups),
            "validation_unique_images": len(validation_groups),
        }

    train_df = pd.DataFrame(train_rows)
    validation_df = pd.DataFrame(validation_rows)

    if train_df.empty or validation_df.empty:
        raise ValueError(
            f"Unable to create non-empty train/validation splits from dataset: {dataset_dir}"
        )

    logger.info(
        "Prepared grouped dataset split: train=%s validation=%s duplicate_summary=%s",
        len(train_df),
        len(validation_df),
        duplicate_summary,
    )
    return train_df, validation_df, duplicate_summary
