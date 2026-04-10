# Pratyush Mishra Chest Cancer Classification

Production-ready chest CT image classification project by **Pratyush Mishra**. The local ML pipeline trains a TensorFlow VGG16 transfer-learning model with DVC reproducibility, and the deployed Vercel API serves a lightweight exported model because TensorFlow exceeds Vercel Python Function storage limits.

Live app: https://chest-cancer-classifier-pratyush.vercel.app

## Architecture

- UI: `templates/index.html`, branded for Pratyush Mishra, supports image upload and prediction.
- API: `app.py`, Flask routes for `/`, `/health`, `/model-info`, `/predict`, and `/train`.
- Local CNN model: `artifacts/training/model.h5`, synced to `model/model.h5` for TensorFlow inference.
- Vercel model: `model/lightweight_model.json`, a trained nearest-centroid image-feature classifier exported from the CT dataset for serverless inference.
- Pipeline: `dvc.yaml` stages for data ingestion, base model preparation, CNN training, evaluation, and lightweight deployment export.
- Logging: rotating backend/API/pipeline logs with timestamp, level, module, trace ID, and `author=Pratyush Mishra`.
- Safety: prediction uploads are decoded into per-request temporary files, validated as images, size-limited, and removed after inference.

## Local Setup

Windows PowerShell:

```powershell
.\scripts\setup.ps1 -PythonPath python
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
chmod +x scripts/setup.sh scripts/run.sh
./scripts/setup.sh python3
source .venv/bin/activate
```

Runtime-only dependencies live in `requirements.txt`. Full ML/DVC/test dependencies live in `requirements-pipeline.txt`.

## Run Locally

One-command app startup:

```powershell
.\scripts\run.ps1
```

Manual commands:

```bash
python app.py
python main.py
dvc repro
python -m pytest
```

Default local URL: `http://localhost:8080`

## API Usage

Health:

```bash
curl https://chest-cancer-classifier-pratyush.vercel.app/health
```

Prediction:

```bash
curl -X POST https://chest-cancer-classifier-pratyush.vercel.app/predict \
  -H "Content-Type: application/json" \
  -d '{"image":"<base64-image>"}'
```

Response example:

```json
[
  {
    "image": "Normal",
    "backend": "lightweight",
    "confidence": 0.0982
  }
]
```

Model info:

```bash
curl https://chest-cancer-classifier-pratyush.vercel.app/model-info
```

Training status:

```bash
curl https://chest-cancer-classifier-pratyush.vercel.app/train
```

`POST /train` runs locally, where TensorFlow/DVC dependencies and writable artifacts are available. On Vercel, it returns `409` with `train_available=false` instead of failing at runtime.

## ML Pipeline

The DVC pipeline runs:

1. Data ingestion from Google Drive or existing `artifacts/data_ingestion/data.zip`.
2. VGG16 base model preparation.
3. One-epoch CNN training using the CT dataset.
4. Evaluation to `scores.json`.
5. Lightweight deployment model export to `model/lightweight_model.json`.

Current verified CNN evaluation:

```json
{
  "loss": 0.11884792894124985,
  "accuracy": 1.0
}
```

Dataset currently contains 343 CT images: 195 adenocarcinoma and 148 normal.
The lightweight Vercel model currently scores 313/343 correct in-sample, about 91.25%, on the available dataset.

## Deployment

Vercel config:

- `.python-version` pins Vercel to Python 3.12.
- `vercel.json` routes all requests to `api/index.py`.
- `.vercelignore` excludes local training artifacts and the large TensorFlow H5 model from serverless upload.
- Vercel runtime uses `USE_LIGHTWEIGHT_MODEL` behavior automatically via the `VERCEL` environment variable.

Production deployment:

```bash
npx vercel@latest deploy --prod --yes
```

Verified live endpoints:

- `GET /health`: `status=ok`, `backend=lightweight`, `author=Pratyush Mishra`.
- `GET /model-info`: reports active model path, backend, and training availability.
- `POST /predict`: returns a classification from `model/lightweight_model.json`.
- `GET /train`: reports training unavailable on Vercel.
- `POST /train`: returns `409` on Vercel by design, because full CNN retraining is local/DVC-only.

## Environment

Copy `.env.example` to `.env` when running locally. Important variables:

- `APP_HOST`, `APP_PORT`
- `MODEL_PATH`
- `LIGHTWEIGHT_MODEL_PATH`
- `USE_LIGHTWEIGHT_MODEL`
- `TRAINED_MODEL_PATH`
- `TRAINING_DATA_PATH`
- `DATA_SOURCE_URL`
- `ENABLE_MLFLOW_LOGGING`
- `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`

## Verification Completed

- `python -m pytest`: 16 passed.
- `python -m pip check`: no broken requirements.
- `dvc repro`: all stages completed and `dvc status` is clean.
- Local TensorFlow API prediction: passed.
- Local lightweight API prediction: passed.
- Vercel production deployment: ready and live.
- Vercel live `/health`, `/model-info`, `/predict`, invalid image handling, and `/train` availability: passed.
- Vercel production error log scan: no errors found after final validation.

## Fixed Integration Gaps

- `/train` no longer fails on Vercel; it clearly reports serverless training is unavailable and remains fully functional locally.
- `/predict` no longer writes all requests to the same image path; each request uses an isolated temporary file.
- Invalid base64 and non-image uploads now return `400` instead of generic server errors.
- The UI now prompts when Predict is clicked before uploading an image and restricts file selection to images.
- CNN training stability was improved with deterministic seeding, Adam, lower learning rate, 3 epochs, and DVC-tracked `RANDOM_SEED`.
- `/model-info` was added so the active backend and model path are transparent.
