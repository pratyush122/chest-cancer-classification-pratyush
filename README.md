# Pratyush Mishra Chest Cancer Classification

Chest CT classification project by **Pratyush Mishra** with a TensorFlow VGG16 transfer-learning pipeline, DVC reproducibility, and a polished Flask + Vercel interface.

Live app: [https://chest-cancer-classifier-pratyush.vercel.app](https://chest-cancer-classifier-pratyush.vercel.app)

## What Changed

- The project is now **deep-learning only**.
- The old handcrafted lightweight classifier has been removed.
- Deployment now uses a **TensorFlow Lite export** of the trained deep model.
- Validation is more honest because exact duplicate images are grouped before train/validation splitting.
- Unsupported or random images are rejected before diagnosis instead of being forced into a cancer class.

## Architecture

- UI: `templates/index.html`
- API: `app.py`
- Training model: `artifacts/training/model.h5`
- Deployment model: `model/model.tflite`
- Input safety profile: `model/inference_profile.json`
- Pipeline stages: data ingestion, base model prep, training, evaluation, TensorFlow Lite export

## Model Type

This project uses **deep learning**, not a classical machine learning classifier.

- Backbone: `VGG16`
- Framework: `TensorFlow / Keras`
- Training style: transfer learning
- Deployment format: `TensorFlow Lite`

## Current Evaluation

Current cleaned validation metrics from `scores.json`:

```json
{
  "loss": 0.8280138373374939,
  "accuracy": 0.86,
  "precision_macro": 0.8055555555555556,
  "recall_macro": 0.9102564102564102,
  "f1_macro": 0.8300145701796989,
  "roc_auc": 0.986013986013986
}
```

Confusion matrix:

```json
[
  [32, 7],
  [0, 11]
]
```

Important note:

- The old `1.0` accuracy number was not trustworthy.
- The dataset contains many duplicate `normal` images.
- The current scores are more legitimate because duplicate leakage is handled before splitting.

## Dataset Notes

- Total images: `343`
- Adenocarcinoma: `195`
- Normal: `148`
- Unique normal images after duplicate grouping: `55`

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

## Run

```bash
python app.py
python main.py
dvc repro
python -m pytest
```

Default local URL: `http://localhost:8080`

## API

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

Model info:

```bash
curl https://chest-cancer-classifier-pratyush.vercel.app/model-info
```

## Environment Variables

- `APP_HOST`
- `APP_PORT`
- `MODEL_PATH`
- `TFLITE_MODEL_PATH`
- `INFERENCE_PROFILE_PATH`
- `USE_TFLITE_MODEL`
- `TRAINED_MODEL_PATH`
- `TRAINING_DATA_PATH`
- `DATA_SOURCE_URL`
- `ENABLE_MLFLOW_LOGGING`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`

## Deployment

Vercel runtime now ships:

- `model/model.tflite`
- `model/inference_profile.json`
- Flask app and source files

Vercel is pinned to **Python 3.12** and uses **LiteRT** (`ai-edge-litert`) for deployment inference.

## Verification

- `python -m pytest`: passed
- `dvc repro tflite_export`: passed
- API rejects non-CT noise image with `422`
- API accepts valid CT uploads
- UI rebuilt with a more professional upload flow, improved hierarchy, and animated About Me section
