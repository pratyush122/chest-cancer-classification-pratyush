import os
import json
from functools import lru_cache

import numpy as np
from PIL import Image

from cnnClassifier import logger
from cnnClassifier.pipeline.modalities import get_modality_config
from cnnClassifier.utils.image_features import (
    INFERENCE_PROFILE_IMAGE_SIZE,
    evaluate_ood_profile,
    extract_image_statistics,
)


class UnsupportedImageError(ValueError):
    pass


@lru_cache(maxsize=2)
def load_inference_profile(model_path: str) -> dict:
    logger.info("Loading inference profile from %s", model_path)
    with open(model_path, "r", encoding="utf-8") as model_file:
        return json.load(model_file)


@lru_cache(maxsize=2)
def load_inference_model(model_path: str):
    from tensorflow.keras.models import load_model

    logger.info("Loading inference model from %s", model_path)
    return load_model(model_path)


@lru_cache(maxsize=2)
def load_tflite_interpreter(model_path: str):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter

    logger.info("Loading TensorFlow Lite model from %s", model_path)
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


class PredictionPipeline:
    def __init__(self, input_image_path: str, modality: str = "chest_ct"):
        self.input_image_path = input_image_path
        self.config = get_modality_config(modality)
        self.model_path = self.config.model_path
        self.tflite_model_path = self.config.tflite_model_path
        self.inference_profile_path = self.config.inference_profile_path
        self.backend = self._select_backend()
        self._image_statistics = None

    def _select_backend(self) -> str:
        use_tflite_env = os.getenv("USE_TFLITE_MODEL", "").lower()
        running_on_vercel = os.getenv("VERCEL", "").lower() in {"1", "true"}
        tflite_available = os.path.exists(self.tflite_model_path)
        tensorflow_available = os.path.exists(self.model_path)

        if use_tflite_env in {"0", "false", "no"}:
            if not tensorflow_available and tflite_available:
                return "tflite"
            return "tensorflow"
        if use_tflite_env in {"1", "true", "yes"}:
            return "tflite"
        if running_on_vercel:
            return "tflite"
        if not tensorflow_available and tflite_available:
            return "tflite"
        return "tensorflow"

    @staticmethod
    def _vgg16_preprocess(image_array: np.ndarray) -> np.ndarray:
        processed = image_array.astype(np.float32).copy()
        processed = processed[:, :, ::-1]
        processed[:, :, 0] -= 103.939
        processed[:, :, 1] -= 116.779
        processed[:, :, 2] -= 123.68
        return processed

    @staticmethod
    def _mobilenet_v2_preprocess(image_array: np.ndarray) -> np.ndarray:
        processed = image_array.astype(np.float32)
        return (processed / 127.5) - 1.0

    def _preprocess_image(self, image_array: np.ndarray) -> np.ndarray:
        if self.config.preprocessing == "mobilenet_v2":
            return self._mobilenet_v2_preprocess(image_array)
        return self._vgg16_preprocess(image_array)

    def _prepare_image_tensor(self) -> np.ndarray:
        from tensorflow.keras.preprocessing import image
        loaded_image = image.load_img(self.input_image_path, target_size=(224, 224))
        image_array = image.img_to_array(loaded_image)
        return np.expand_dims(self._preprocess_image(image_array), axis=0)

    def _prepare_tflite_tensor(self) -> np.ndarray:
        with Image.open(self.input_image_path) as uploaded_image:
            image_array = np.asarray(
                uploaded_image.convert("RGB").resize((224, 224)),
                dtype=np.float32,
            )
        return np.expand_dims(self._preprocess_image(image_array), axis=0)

    def _image_feature_stats(self) -> dict[str, float]:
        if self._image_statistics is None:
            self._image_statistics = extract_image_statistics(
                self.input_image_path,
                image_size=INFERENCE_PROFILE_IMAGE_SIZE,
            )
        return self._image_statistics

    def _ensure_supported_image_domain(self):
        if not os.path.exists(self.inference_profile_path):
            logger.warning(
                "Skipping image-domain validation because inference profile is missing at %s",
                self.inference_profile_path,
            )
            return

        ood_profile = load_inference_profile(self.inference_profile_path).get("ood_profile")
        if not ood_profile:
            return

        supported, anomaly_score, violations = evaluate_ood_profile(
            self._image_feature_stats(),
            ood_profile,
        )
        if supported:
            logger.info("Image-domain validation passed with anomaly_score=%.4f", anomaly_score)
            return

        logger.warning(
            "Rejected unsupported image domain with anomaly_score=%.4f violations=%s",
            anomaly_score,
            violations,
        )
        raise UnsupportedImageError(self.config.unsupported_image_message)

    def _predict_with_tflite_model(self):
        if not os.path.exists(self.tflite_model_path):
            raise FileNotFoundError(f"TensorFlow Lite model file not found at {self.tflite_model_path}")

        interpreter, input_details, output_details = load_tflite_interpreter(self.tflite_model_path)
        input_tensor = self._prepare_tflite_tensor().astype(input_details[0]["dtype"])
        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()
        probabilities = interpreter.get_tensor(output_details[0]["index"])[0]
        predicted_index = int(np.argmax(probabilities))
        prediction_label = self.config.labels[predicted_index] if predicted_index < len(self.config.labels) else "Unknown"
        confidence = float(probabilities[predicted_index])
        logger.info(
            "TensorFlow Lite prediction completed: modality=%s label=%s, class_index=%s, confidence=%.4f",
            self.config.key,
            prediction_label,
            predicted_index,
            confidence,
        )
        return [
            {
                "image": prediction_label,
                "class_index": predicted_index,
                "backend": "tflite",
                "modality": self.config.key,
                "confidence": round(confidence, 4),
                "probabilities": {
                    self.config.labels[index]: round(float(probability), 4)
                    for index, probability in enumerate(probabilities)
                },
                "supported_image": True,
            }
        ]

    def _predict_with_tensorflow_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        model = load_inference_model(self.model_path)
        input_tensor = self._prepare_image_tensor()
        probabilities = model.predict(input_tensor, verbose=0)
        predicted_index = int(np.argmax(probabilities, axis=1)[0])
        prediction_label = self.config.labels[predicted_index] if predicted_index < len(self.config.labels) else "Unknown"
        confidence = float(probabilities[0][predicted_index])

        logger.info(
            "Prediction completed: modality=%s label=%s, class_index=%s, confidence=%.4f",
            self.config.key,
            prediction_label,
            predicted_index,
            confidence,
        )
        return [
            {
                "image": prediction_label,
                "class_index": predicted_index,
                "backend": "tensorflow",
                "modality": self.config.key,
                "confidence": round(confidence, 4),
                "probabilities": {
                    self.config.labels[index]: round(float(probability), 4)
                    for index, probability in enumerate(probabilities[0])
                },
                "supported_image": True,
            }
        ]

    def predict(self):
        self._ensure_supported_image_domain()
        if self.backend == "tflite":
            return self._predict_with_tflite_model()
        return self._predict_with_tensorflow_model()
