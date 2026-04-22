import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, g, jsonify, render_template, request
from flask_cors import CORS, cross_origin
from PIL import Image, UnidentifiedImageError

from cnnClassifier import PROJECT_AUTHOR, logger
from cnnClassifier.pipeline.modalities import (
    get_default_modality_key,
    get_modality_config,
    list_modality_configs,
)
from cnnClassifier.pipeline.prediction import PredictionPipeline, UnsupportedImageError
from cnnClassifier.utils.common import ImageDecodeError, decodeImage

load_dotenv()

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8080"))
DEFAULT_INPUT_IMAGE_NAME = "/tmp/inputImage.jpg" if os.getenv("VERCEL") else "inputImage.jpg"
INPUT_IMAGE_NAME = os.getenv("INPUT_IMAGE_NAME", DEFAULT_INPUT_IMAGE_NAME)
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", "5242880"))
RUNNING_ON_VERCEL = os.getenv("VERCEL", "").lower() in {"1", "true"}

app = Flask(__name__)
CORS(app)


class InferenceService:
    def __init__(self):
        self.input_image_name = INPUT_IMAGE_NAME
        self.default_modality = get_default_modality_key()
        self.predictor = PredictionPipeline(self.input_image_name, modality=self.default_modality)

    def create_predictor(self, input_image_path: str, modality: str = "chest_ct") -> PredictionPipeline:
        return PredictionPipeline(input_image_path, modality=modality)


inference_service = InferenceService()


def active_backend() -> str:
    use_tflite_env = os.getenv("USE_TFLITE_MODEL", "").lower()
    running_on_vercel = os.getenv("VERCEL", "").lower() in {"1", "true"}

    if use_tflite_env in {"0", "false", "no"}:
        return "tensorflow"
    if use_tflite_env in {"1", "true", "yes"}:
        return "tflite"
    if running_on_vercel:
        return "tflite"
    return "tensorflow"


def public_modality_payload() -> list[dict]:
    backend = active_backend()
    return [config.to_public_dict(backend) for config in list_modality_configs()]


@app.before_request
def start_request_trace():
    g.trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
    g.request_started_at = time.perf_counter()
    logger.info(
        "API request started method=%s path=%s remote_addr=%s",
        request.method,
        request.path,
        request.headers.get("X-Forwarded-For", request.remote_addr),
        extra={"trace_id": g.trace_id},
    )


@app.after_request
def finish_request_trace(response):
    duration_ms = round((time.perf_counter() - g.request_started_at) * 1000, 2)
    response.headers["X-Trace-ID"] = g.trace_id
    response.headers["X-Author"] = PROJECT_AUTHOR
    logger.info(
        "API request completed method=%s path=%s status=%s duration_ms=%s",
        request.method,
        request.path,
        response.status_code,
        duration_ms,
        extra={"trace_id": g.trace_id},
    )
    return response


@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
@cross_origin()
def health_route():
    predictor = inference_service.predictor
    model_path = (
        predictor.tflite_model_path
        if predictor.backend == "tflite"
        else predictor.model_path
    )
    return jsonify(
        {
            "status": "ok",
            "service": "chest-cancer-classifier",
            "author": PROJECT_AUTHOR,
            "backend": predictor.backend,
            "model_path": model_path,
            "model_available": os.path.exists(model_path),
            "train_available": not RUNNING_ON_VERCEL,
            "default_modality": inference_service.default_modality,
            "modalities": public_modality_payload(),
        }
    )


@app.route("/modalities", methods=["GET"])
@cross_origin()
def modalities_route():
    return jsonify(
        {
            "author": PROJECT_AUTHOR,
            "default_modality": inference_service.default_modality,
            "modalities": public_modality_payload(),
        }
    )


@app.route("/model-info", methods=["GET"])
@cross_origin()
def model_info_route():
    payload = {
        "author": PROJECT_AUTHOR,
        "default_modality": inference_service.default_modality,
        "modalities": {item["key"]: item for item in public_modality_payload()},
        "train_available": not RUNNING_ON_VERCEL,
    }
    return jsonify(payload)


@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def train_route():
    if request.method == "GET":
        return jsonify(
            {
                "status": "available" if not RUNNING_ON_VERCEL else "unavailable",
                "author": PROJECT_AUTHOR,
                "train_available": not RUNNING_ON_VERCEL,
                "message": (
                    "POST /train runs the local multi-modal training pipeline for chest CT and ECG models."
                    if not RUNNING_ON_VERCEL
                    else "Training is disabled on Vercel serverless runtime; run python main.py locally."
                ),
            }
        )

    if RUNNING_ON_VERCEL:
        return (
            jsonify(
                {
                    "status": "unavailable",
                    "author": PROJECT_AUTHOR,
                    "train_available": False,
                    "message": "Training is disabled on Vercel serverless runtime; run python main.py locally.",
                }
            ),
            409,
        )

    try:
        logger.info("Training pipeline requested.", extra={"trace_id": g.trace_id})
        # Use the current interpreter so this works in active virtual environments.
        subprocess.run([sys.executable, "main.py"], check=True)
    except subprocess.CalledProcessError as error:
        logger.exception("Training pipeline failed.", extra={"trace_id": g.trace_id})
        return jsonify({"status": "failed", "message": str(error)}), 500
    return jsonify(
        {
            "status": "success",
            "message": "Training completed.",
            "author": PROJECT_AUTHOR,
        }
    )


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict_route():
    request_data = request.get_json(silent=True) or {}
    image_base64 = request_data.get("image")
    requested_modality = request_data.get("modality", inference_service.default_modality)

    if not image_base64:
        return jsonify({"error": "Missing 'image' key in JSON payload."}), 400

    try:
        modality_config = get_modality_config(requested_modality)
    except ValueError:
        return jsonify({"error": "Unsupported modality."}), 400

    temp_image_path = Path(tempfile.gettempdir()) / f"chest_classifier_{g.trace_id}.jpg"
    try:
        decodeImage(image_base64, temp_image_path, max_bytes=MAX_IMAGE_BYTES)
        try:
            with Image.open(temp_image_path) as uploaded_image:
                uploaded_image.verify()
        except (UnidentifiedImageError, OSError) as error:
            raise ImageDecodeError("Decoded payload is not a supported image file.") from error
        predictor = inference_service.create_predictor(str(temp_image_path), modality=modality_config.key)
        prediction_result = predictor.predict()
    except ImageDecodeError as error:
        logger.warning(
            "Invalid prediction image payload: %s",
            error,
            extra={"trace_id": g.trace_id},
        )
        return jsonify({"error": "Invalid image payload.", "message": str(error)}), 400
    except UnsupportedImageError as error:
        logger.warning(
            "Unsupported prediction image: %s",
            error,
            extra={"trace_id": g.trace_id},
        )
        return jsonify({"error": "Unsupported image domain.", "message": str(error)}), 422
    except Exception as error:
        logger.exception("Prediction failed.", extra={"trace_id": g.trace_id})
        return jsonify({"error": "Prediction failed.", "message": str(error)}), 500
    finally:
        try:
            temp_image_path.unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "Could not remove temporary image file: %s",
                temp_image_path,
                extra={"trace_id": g.trace_id},
            )

    return jsonify(prediction_result)


if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT)
