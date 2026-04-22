# Pratyush Mishra Medical Image Classifier

Combined final-year deep-learning project for:

- Chest CT scan classification for adenocarcinoma vs normal
- ECG image classification for cardiovascular condition screening

Live app: [https://chest-cancer-classifier-pratyush.vercel.app](https://chest-cancer-classifier-pratyush.vercel.app)

## Project Summary

This repository now ships as a single multi-modal medical-imaging demo. The same Flask + Vercel app lets the user choose between:

- `Chest CT Scan`
- `ECG Image`

The backend routes each upload to the correct deep-learning model, class labels, TensorFlow Lite artifact, and image-domain safety profile.

## Modalities

### 1. Chest CT Scan

- Task: `Adenocarcinoma Cancer` vs `Normal`
- Backbone: `VGG16`
- Local trained model: `artifacts/training/model.h5`
- Deployment model: `model/model.tflite`
- Input profile: `model/inference_profile.json`

### 2. ECG Image

- Task: `Abnormal heartbeat`, `History of MI`, `Normal Person`
- Backbone: `MobileNetV2`
- Dataset source: `https://zenodo.org/api/records/14767442/files/archive%20(10).zip/content`
- Local trained model: `artifacts/ecg_training/model.h5`
- Deployment model: `model/ecg_model.tflite`
- Input profile: `model/ecg_inference_profile.json`
- Metadata: `model/ecg_metadata.json`

## Current Scores

### Chest CT

Metrics from `scores.json`:

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

### ECG

Metrics from `ecg_scores.json`:

```json
{
  "loss": 0.4825614392757416,
  "accuracy": 0.7865168539325843,
  "precision_macro": 0.7281842818428185,
  "recall_macro": 0.7338298955946015,
  "f1_macro": 0.7301844875881387
}
```

Important note:

- The ECG dataset has only `89` validation images after duplicate-aware grouping.
- `History of MI` is the hardest class because it has the smallest unique-image count.
- These are honest validation metrics from the actual trained artifacts included in this repo.

## Architecture

- UI: `templates/index.html`
- API: `app.py`
- Vercel entry: `api/index.py`
- Shared modality registry: `src/cnnClassifier/pipeline/modalities.py`
- Shared inference path: `src/cnnClassifier/pipeline/prediction.py`
- Chest CT training pipeline: `main.py` + DVC chest stages
- ECG training script: `scripts/train_ecg_model.py`

## What Changed

- Added ECG image classification as a second diagnostic modality
- Added a shared modality registry for model paths, labels, and image-domain rules
- Updated the frontend to let the user choose `Chest CT Scan` or `ECG Image`
- Added a new ECG training/export pipeline that:
  - downloads the ECG dataset into the project on `D:`
  - trains the ECG image model locally
  - exports `model/ecg_model.tflite`
  - builds `model/ecg_inference_profile.json`
  - saves `ecg_scores.json`
- Updated `main.py` so local full training runs both the chest CT pipeline and the ECG pipeline
- Updated Vercel packaging to ship both deployed TFLite models

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

## Run The App

```powershell
python app.py
```

Open:

- [http://localhost:8080](http://localhost:8080)

## Train Both Models Locally

```powershell
python main.py
```

What `python main.py` does now:

1. Runs the chest CT ingestion stage
2. Prepares the chest CT base model
3. Trains the chest CT model
4. Evaluates the chest CT model
5. Trains the ECG image model
6. Exports ECG deployment artifacts

## Train Only The ECG Model

```powershell
python scripts/train_ecg_model.py
```

Dataset download location:

- `datasets/ecg/`

This was intentionally kept inside the project on `D:` to avoid heavy downloads on `C:`.

## API

### Health

```bash
curl http://localhost:8080/health
```

### Modalities

```bash
curl http://localhost:8080/modalities
```

### Model Info

```bash
curl http://localhost:8080/model-info
```

### Predict

Chest CT:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d "{\"modality\":\"chest_ct\",\"image\":\"<base64-image>\"}"
```

ECG:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d "{\"modality\":\"ecg\",\"image\":\"<base64-image>\"}"
```

## Environment Variables

- `APP_HOST`
- `APP_PORT`
- `MODEL_PATH`
- `TFLITE_MODEL_PATH`
- `INFERENCE_PROFILE_PATH`
- `ECG_MODEL_PATH`
- `ECG_TFLITE_MODEL_PATH`
- `ECG_INFERENCE_PROFILE_PATH`
- `USE_TFLITE_MODEL`
- `TRAINED_MODEL_PATH`
- `TRAINING_DATA_PATH`
- `DATA_SOURCE_URL`
- `ENABLE_MLFLOW_LOGGING`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`

## Vercel Deployment

Vercel ships the deployable inference artifacts for both modalities:

- `model/model.tflite`
- `model/inference_profile.json`
- `model/ecg_model.tflite`
- `model/ecg_inference_profile.json`
- `model/ecg_metadata.json`

Training is disabled on Vercel serverless runtime. Training must be done locally with:

```powershell
python main.py
```

After local training, commit the generated deployable ECG artifacts and redeploy.

## MLflow

The repository still supports MLflow evaluation logging when enabled:

```powershell
$env:ENABLE_MLFLOW_LOGGING="true"
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
python -m cnnClassifier.pipeline.stage_04_model_evaluation
```

That logs the chest CT evaluation run. The ECG training pipeline currently saves its metrics to `ecg_scores.json`.

## Verification

Recommended checks:

```powershell
python -m pytest
python app.py
python scripts/train_ecg_model.py
```

Manual checks:

- choose `Chest CT Scan` in the UI and run a prediction
- choose `ECG Image` in the UI and run a prediction
- verify `http://localhost:8080/modalities` returns both modalities
- verify `http://localhost:8080/model-info` exposes both deployed models
