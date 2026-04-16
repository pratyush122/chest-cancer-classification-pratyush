import json
from pathlib import Path

import tensorflow as tf

from cnnClassifier.utils.data_utils import SUPPORTED_IMAGE_EXTENSIONS
from cnnClassifier.utils.image_features import build_ood_profile, extract_image_statistics

AUTHOR = "Pratyush Mishra"
DATA_DIR = Path("artifacts/data_ingestion/Chest-CT-Scan-data")
KERAS_MODEL_PATH = Path("artifacts/training/model.h5")
TFLITE_MODEL_OUTPUT_PATH = Path("model/model.tflite")
PROFILE_OUTPUT_PATH = Path("model/inference_profile.json")


def main() -> None:
    if not KERAS_MODEL_PATH.exists():
        raise FileNotFoundError(f"Keras model not found at {KERAS_MODEL_PATH}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found at {DATA_DIR}")

    model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    feature_dicts = []
    dataset_counts = {}
    for class_dir in sorted(path for path in DATA_DIR.iterdir() if path.is_dir()):
        image_paths = [
            image_path
            for image_path in sorted(class_dir.iterdir())
            if image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        dataset_counts[class_dir.name] = len(image_paths)
        for image_path in image_paths:
            feature_dicts.append(extract_image_statistics(image_path))

    if not feature_dicts:
        raise ValueError(f"No supported images found in {DATA_DIR}")

    TFLITE_MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TFLITE_MODEL_OUTPUT_PATH.write_bytes(tflite_model)
    PROFILE_OUTPUT_PATH.write_text(
        json.dumps(
            {
                "author": AUTHOR,
                "model_type": "tensorflow_lite_vgg16",
                "source_model_path": str(KERAS_MODEL_PATH),
                "dataset_counts": dataset_counts,
                "ood_profile": build_ood_profile(feature_dicts),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved TensorFlow Lite model to {TFLITE_MODEL_OUTPUT_PATH}")
    print(f"Saved inference profile to {PROFILE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
