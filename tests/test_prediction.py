import numpy as np
from PIL import Image

from cnnClassifier.pipeline import prediction


class FakeModel:
    def predict(self, input_tensor, verbose=0):
        assert input_tensor.shape == (1, 224, 224, 3)
        assert verbose == 0
        return np.array([[0.9, 0.1]])


def test_prediction_pipeline_normalizes_and_labels(monkeypatch, tmp_path):
    image_path = tmp_path / "scan.png"
    model_path = tmp_path / "model.h5"
    Image.new("RGB", (32, 32), color=(255, 255, 255)).save(image_path)
    model_path.write_bytes(b"fake-model")

    prediction.load_inference_model.cache_clear()
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setattr(prediction, "load_inference_model", lambda model_path: FakeModel())

    result = prediction.PredictionPipeline(str(image_path)).predict()

    assert result == [
        {"image": "Adenocarcinoma Cancer", "class_index": 0, "backend": "tensorflow"}
    ]
