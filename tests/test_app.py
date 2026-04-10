import base64
import io
import subprocess

from PIL import Image

import app as app_module


def _encoded_png() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color=(255, 255, 255)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_home_contains_pratyush_mishra_branding():
    client = app_module.app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"Pratyush Mishra" in response.data


def test_health_reports_model_availability():
    client = app_module.app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["author"] == "Pratyush Mishra"
    assert payload["model_available"] is True
    assert "train_available" in payload


def test_model_info_reports_active_backend():
    client = app_module.app.test_client()

    response = client.get("/model-info")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["author"] == "Pratyush Mishra"
    assert payload["model_available"] is True
    assert payload["backend"] in {"tensorflow", "lightweight"}


def test_predict_requires_image_payload():
    client = app_module.app.test_client()

    response = client.post("/predict", json={})

    assert response.status_code == 400
    assert response.get_json()["error"] == "Missing 'image' key in JSON payload."


def test_predict_returns_model_result(monkeypatch, tmp_path):
    created_paths = []

    class FakePredictor:
        def __init__(self, input_image_path):
            created_paths.append(input_image_path)

        def predict(self):
            return [{"image": "Normal", "class_index": 1}]

    monkeypatch.setattr(
        app_module.inference_service,
        "create_predictor",
        lambda input_image_path: FakePredictor(input_image_path),
    )
    client = app_module.app.test_client()

    response = client.post("/predict", json={"image": _encoded_png()})

    assert response.status_code == 200
    assert response.get_json() == [{"image": "Normal", "class_index": 1}]
    assert len(created_paths) == 1


def test_predict_rejects_invalid_base64_payload():
    client = app_module.app.test_client()

    response = client.post("/predict", json={"image": "not base64 %%%"})

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid image payload."


def test_predict_rejects_non_image_payload():
    client = app_module.app.test_client()
    encoded = base64.b64encode(b"not actually an image").decode("ascii")

    response = client.post("/predict", json={"image": encoded})

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid image payload."


def test_train_get_reports_availability():
    client = app_module.app.test_client()

    response = client.get("/train")

    assert response.status_code == 200
    assert response.get_json()["author"] == "Pratyush Mishra"


def test_train_route_runs_main_with_current_interpreter(monkeypatch):
    calls = []

    def fake_run(command, check):
        calls.append((command, check))
        return None

    monkeypatch.setattr(app_module.subprocess, "run", fake_run)
    client = app_module.app.test_client()

    response = client.post("/train")

    assert response.status_code == 200
    assert response.get_json()["author"] == "Pratyush Mishra"
    assert calls == [([app_module.sys.executable, "main.py"], True)]


def test_train_route_reports_subprocess_failure(monkeypatch):
    def fake_run(command, check):
        raise subprocess.CalledProcessError(returncode=1, cmd=command)

    monkeypatch.setattr(app_module.subprocess, "run", fake_run)
    client = app_module.app.test_client()

    response = client.post("/train")

    assert response.status_code == 500
    assert response.get_json()["status"] == "failed"


def test_train_route_reports_vercel_runtime_unavailable(monkeypatch):
    monkeypatch.setattr(app_module, "RUNNING_ON_VERCEL", True)
    client = app_module.app.test_client()

    response = client.post("/train")

    assert response.status_code == 409
    payload = response.get_json()
    assert payload["status"] == "unavailable"
    assert payload["train_available"] is False
