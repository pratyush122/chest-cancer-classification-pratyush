import json
from pathlib import Path

import numpy as np
from PIL import Image

AUTHOR = "Pratyush Mishra"
DATA_DIR = Path("artifacts/data_ingestion/Chest-CT-Scan-data")
OUTPUT_PATH = Path("model/lightweight_model.json")
IMAGE_SIZE = 128


def extract_features(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as opened_image:
        image_array = np.asarray(
            opened_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)),
            dtype=np.float32,
        ) / 255.0

    grayscale = image_array.mean(axis=2)
    dx = np.abs(np.diff(grayscale, axis=1)).mean()
    dy = np.abs(np.diff(grayscale, axis=0)).mean()
    return np.array(
        [
            grayscale.mean(),
            grayscale.std(),
            np.percentile(grayscale, 10),
            np.percentile(grayscale, 90),
            image_array[:, :, 0].mean(),
            image_array[:, :, 1].mean(),
            image_array[:, :, 2].mean(),
            image_array[:, :, 0].std(),
            image_array[:, :, 1].std(),
            image_array[:, :, 2].std(),
            dx,
            dy,
        ],
        dtype=np.float32,
    )


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

    samples = []
    labels = []
    label_map = {
        "adenocarcinoma": "Adenocarcinoma Cancer",
        "normal": "Normal",
    }

    for class_dir in sorted(DATA_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        label = label_map.get(class_dir.name, class_dir.name)
        for image_path in sorted(class_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            samples.append(extract_features(image_path))
            labels.append(label)

    if not samples:
        raise ValueError(f"No images found in {DATA_DIR}")

    feature_matrix = np.vstack(samples)
    feature_mean = feature_matrix.mean(axis=0)
    feature_std = feature_matrix.std(axis=0) + 1e-8
    normalized = (feature_matrix - feature_mean) / feature_std

    centroids = {}
    counts = {}
    for label in sorted(set(labels)):
        indices = [index for index, sample_label in enumerate(labels) if sample_label == label]
        centroids[label] = normalized[indices].mean(axis=0).tolist()
        counts[label] = len(indices)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "author": AUTHOR,
                "model_type": "nearest_centroid_image_features",
                "image_size": IMAGE_SIZE,
                "feature_mean": feature_mean.tolist(),
                "feature_std": feature_std.tolist(),
                "centroids": centroids,
                "class_counts": counts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved lightweight deployment model to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
