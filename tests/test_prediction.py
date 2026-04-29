import numpy as np
import pytest
from PIL import Image

from cnnClassifier.pipeline import prediction


class FakeModel:
    def predict(self, input_tensor, verbose=0):
        assert input_tensor.shape == (1, 224, 224, 3)
        assert verbose == 0
        assert float(input_tensor[0, 0, 0, 0]) == pytest.approx(16.061, abs=1e-3)
        return np.array([[0.9, 0.1]])


class FakeEcgModel:
    def predict(self, input_tensor, verbose=0):
        assert input_tensor.shape == (1, 224, 224, 3)
        assert verbose == 0
        assert float(input_tensor[0, 0, 0, 0]) == pytest.approx(0.9216, abs=1e-3)
        return np.array([[0.1, 0.8, 0.1]])


def test_prediction_pipeline_normalizes_and_labels(monkeypatch, tmp_path):
    image_path = tmp_path / "scan.png"
    model_path = tmp_path / "model.h5"
    Image.new("RGB", (32, 32), color=(120, 120, 120)).save(image_path)
    model_path.write_bytes(b"fake-model")
    inference_profile_path = tmp_path / "inference_profile.json"
    inference_profile_path.write_text(
        """
{
  "ood_profile": {
    "feature_names": ["grayscale_mean", "grayscale_std", "grayscale_p10", "grayscale_p90", "red_mean", "green_mean", "blue_mean", "red_std", "green_std", "blue_std", "edge_horizontal", "edge_vertical", "dark_pixel_ratio", "bright_pixel_ratio", "border_dark_ratio", "center_mean", "center_std", "channel_delta", "entropy"],
    "feature_mean": [0.47, 0.02, 0.47, 0.47, 0.47, 0.47, 0.47, 0.02, 0.02, 0.02, 0.02, 0.02, 0.0, 0.0, 0.0, 0.47, 0.02, 0.0, 0.1],
    "feature_std": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    "score_threshold": 20.0,
    "feature_bounds": {
      "grayscale_std": {"min": 0.0, "max": 1.0},
      "edge_horizontal": {"min": 0.0, "max": 1.0},
      "edge_vertical": {"min": 0.0, "max": 1.0},
      "entropy": {"min": -1.0, "max": 10.0},
      "channel_delta": {"min": 0.0, "max": 1.0}
    }
  }
}
        """,
        encoding="utf-8",
    )

    prediction.load_inference_model.cache_clear()
    prediction.load_inference_profile.cache_clear()
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("INFERENCE_PROFILE_PATH", str(inference_profile_path))
    monkeypatch.setenv("USE_TFLITE_MODEL", "false")
    monkeypatch.setattr(prediction, "load_inference_model", lambda model_path: FakeModel())

    result = prediction.PredictionPipeline(str(image_path)).predict()

    assert result == [
        {
            "image": "Adenocarcinoma Cancer",
            "class_index": 0,
            "backend": "tensorflow",
            "modality": "chest_ct",
            "confidence": 0.9,
            "probabilities": {
                "Adenocarcinoma Cancer": 0.9,
                "Normal": 0.1,
            },
            "supported_image": True,
        }
    ]


def test_prediction_pipeline_supports_ecg_modality(monkeypatch, tmp_path):
    image_path = tmp_path / "ecg.png"
    model_path = tmp_path / "ecg_model.h5"
    Image.new("RGB", (32, 32), color=(245, 245, 245)).save(image_path)
    model_path.write_bytes(b"fake-model")
    inference_profile_path = tmp_path / "ecg_inference_profile.json"
    inference_profile_path.write_text(
        """
{
  "ood_profile": {
    "feature_names": ["grayscale_mean", "grayscale_std", "grayscale_p10", "grayscale_p90", "red_mean", "green_mean", "blue_mean", "red_std", "green_std", "blue_std", "edge_horizontal", "edge_vertical", "dark_pixel_ratio", "bright_pixel_ratio", "border_dark_ratio", "center_mean", "center_std", "channel_delta", "entropy"],
    "feature_mean": [0.9, 0.03, 0.85, 0.95, 0.9, 0.9, 0.9, 0.03, 0.03, 0.03, 0.01, 0.01, 0.0, 1.0, 0.0, 0.9, 0.03, 0.0, 0.5],
    "feature_std": [0.2, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
    "score_threshold": 20.0,
    "feature_bounds": {
      "grayscale_std": {"min": 0.0, "max": 1.0},
      "edge_horizontal": {"min": 0.0, "max": 1.0},
      "edge_vertical": {"min": 0.0, "max": 1.0},
      "entropy": {"min": -1.0, "max": 10.0},
      "channel_delta": {"min": 0.0, "max": 1.0}
    }
  }
}
        """,
        encoding="utf-8",
    )

    prediction.load_inference_model.cache_clear()
    prediction.load_inference_profile.cache_clear()
    monkeypatch.setenv("ECG_MODEL_PATH", str(model_path))
    monkeypatch.setenv("ECG_INFERENCE_PROFILE_PATH", str(inference_profile_path))
    monkeypatch.setenv("USE_TFLITE_MODEL", "false")
    monkeypatch.setattr(prediction, "load_inference_model", lambda model_path: FakeEcgModel())

    result = prediction.PredictionPipeline(str(image_path), modality="ecg").predict()

    assert result == [
        {
            "image": "History of MI",
            "class_index": 1,
            "backend": "tensorflow",
            "modality": "ecg",
            "confidence": 0.8,
            "probabilities": {
                "Abnormal heartbeat": 0.1,
                "History of MI": 0.8,
                "Normal Person": 0.1,
            },
            "supported_image": True,
        }
    ]


def test_ecg_tensor_preparation_matches_tflite_path(tmp_path):
    image_path = tmp_path / "ecg_gradient.png"
    gradient = np.tile(np.linspace(220, 255, 64, dtype=np.uint8), (64, 1))
    gradient_rgb = np.stack([gradient, gradient, gradient], axis=2)
    Image.fromarray(gradient_rgb).save(image_path)

    pipeline = prediction.PredictionPipeline(str(image_path), modality="ecg")

    image_tensor = pipeline._prepare_image_tensor()
    tflite_tensor = pipeline._prepare_tflite_tensor()

    assert image_tensor.shape == (1, 224, 224, 3)
    assert tflite_tensor.shape == (1, 224, 224, 3)
    np.testing.assert_allclose(image_tensor, tflite_tensor, atol=1e-6)


def test_prediction_pipeline_rejects_unknown_modality(tmp_path):
    image_path = tmp_path / "scan.png"
    Image.new("RGB", (32, 32), color=(120, 120, 120)).save(image_path)

    with pytest.raises(ValueError):
        prediction.PredictionPipeline(str(image_path), modality="unknown")


def test_prediction_pipeline_rejects_non_ct_images(monkeypatch, tmp_path):
    image_path = tmp_path / "noise.png"
    noise = np.random.default_rng(0).integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(noise).save(image_path)
    inference_profile_path = tmp_path / "inference_profile.json"
    inference_profile_path.write_text(
        """
{
  "ood_profile": {
    "feature_names": ["grayscale_mean", "grayscale_std", "grayscale_p10", "grayscale_p90", "red_mean", "green_mean", "blue_mean", "red_std", "green_std", "blue_std", "edge_horizontal", "edge_vertical", "dark_pixel_ratio", "bright_pixel_ratio", "border_dark_ratio", "center_mean", "center_std", "channel_delta", "entropy"],
    "feature_mean": [0.37, 0.25, 0.06, 0.67, 0.37, 0.37, 0.37, 0.25, 0.25, 0.25, 0.04, 0.03, 0.25, 0.01, 0.75, 0.35, 0.22, 0.0, 2.0],
    "feature_std": [0.13, 0.05, 0.09, 0.19, 0.13, 0.13, 0.13, 0.05, 0.05, 0.05, 0.01, 0.01, 0.2, 0.2, 0.3, 0.2, 0.1, 0.01, 0.2],
    "score_threshold": 10.0,
    "feature_bounds": {
      "grayscale_std": {"min": 0.1, "max": 0.4},
      "edge_horizontal": {"min": 0.01, "max": 0.08},
      "edge_vertical": {"min": 0.01, "max": 0.08},
      "entropy": {"min": 1.3, "max": 2.45},
      "channel_delta": {"min": 0.0, "max": 0.01}
    }
  }
}
        """,
        encoding="utf-8",
    )

    prediction.load_inference_profile.cache_clear()
    monkeypatch.setenv("INFERENCE_PROFILE_PATH", str(inference_profile_path))

    with pytest.raises(prediction.UnsupportedImageError):
        prediction.PredictionPipeline(str(image_path)).predict()
