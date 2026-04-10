import os
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from cnnClassifier import logger


CLASS_LABELS = {
    0: "Adenocarcinoma Cancer",
    1: "Normal",
}


@lru_cache(maxsize=2)
def load_lightweight_model(model_path: str) -> dict:
    logger.info("Loading lightweight inference model from %s", model_path)
    with open(model_path, "r", encoding="utf-8") as model_file:
        return json.load(model_file)


@lru_cache(maxsize=2)
def load_inference_model(model_path: str):
    from tensorflow.keras.models import load_model

    logger.info("Loading inference model from %s", model_path)
    return load_model(model_path)


class PredictionPipeline:
    def __init__(self, input_image_path: str):
        self.input_image_path = input_image_path
        self.model_path = os.getenv("MODEL_PATH", os.path.join("model", "model.h5"))
        self.lightweight_model_path = os.getenv(
            "LIGHTWEIGHT_MODEL_PATH", os.path.join("model", "lightweight_model.json")
        )
        self.backend = self._select_backend()

    def _select_backend(self) -> str:
        use_lightweight = os.getenv("USE_LIGHTWEIGHT_MODEL", "").lower() == "true"
        running_on_vercel = os.getenv("VERCEL", "").lower() in {"1", "true"}
        model_is_json = self.model_path.lower().endswith(".json")

        if use_lightweight or running_on_vercel or model_is_json:
            return "lightweight"
        return "tensorflow"

    def _prepare_image_tensor(self) -> np.ndarray:
        from tensorflow.keras.preprocessing import image

        loaded_image = image.load_img(self.input_image_path, target_size=(224, 224))
        image_array = image.img_to_array(loaded_image)
        normalized_array = image_array / 255.0
        return np.expand_dims(normalized_array, axis=0)

    @staticmethod
    def _extract_lightweight_features(input_image_path: str, image_size: int) -> np.ndarray:
        with Image.open(input_image_path) as opened_image:
            image_array = np.asarray(
                opened_image.convert("RGB").resize((image_size, image_size)),
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

    def _predict_with_lightweight_model(self):
        if not os.path.exists(self.lightweight_model_path):
            raise FileNotFoundError(f"Lightweight model file not found at {self.lightweight_model_path}")

        model = load_lightweight_model(self.lightweight_model_path)
        feature_vector = self._extract_lightweight_features(
            self.input_image_path,
            int(model.get("image_size", 128)),
        )
        feature_mean = np.asarray(model["feature_mean"], dtype=np.float32)
        feature_std = np.asarray(model["feature_std"], dtype=np.float32)
        normalized_features = (feature_vector - feature_mean) / feature_std

        class_distances = {}
        for label, centroid in model["centroids"].items():
            centroid_array = np.asarray(centroid, dtype=np.float32)
            class_distances[label] = float(np.linalg.norm(normalized_features - centroid_array))

        prediction_label = min(class_distances, key=class_distances.get)
        confidence = 1.0 / (1.0 + class_distances[prediction_label])
        logger.info(
            "Lightweight prediction completed: label=%s, confidence=%.4f",
            prediction_label,
            confidence,
        )
        return [
            {
                "image": prediction_label,
                "backend": "lightweight",
                "confidence": round(confidence, 4),
            }
        ]

    def _predict_with_tensorflow_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        model = load_inference_model(self.model_path)
        input_tensor = self._prepare_image_tensor()
        probabilities = model.predict(input_tensor, verbose=0)
        predicted_index = int(np.argmax(probabilities, axis=1)[0])
        prediction_label = CLASS_LABELS.get(predicted_index, "Unknown")

        logger.info(
            "Prediction completed: label=%s, class_index=%s",
            prediction_label,
            predicted_index,
        )
        return [{"image": prediction_label, "class_index": predicted_index, "backend": "tensorflow"}]

    def predict(self):
        if self.backend == "lightweight":
            return self._predict_with_lightweight_model()
        return self._predict_with_tensorflow_model()
