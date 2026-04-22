import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModalityConfig:
    key: str
    display_name: str
    upload_title: str
    upload_hint: str
    summary: str
    labels: tuple[str, ...]
    preprocessing: str
    model_path: str
    tflite_model_path: str
    inference_profile_path: str
    unsupported_image_message: str

    def to_public_dict(self, backend: str) -> dict:
        model_path = self.tflite_model_path if backend == "tflite" else self.model_path
        return {
            "key": self.key,
            "display_name": self.display_name,
            "upload_title": self.upload_title,
            "upload_hint": self.upload_hint,
            "summary": self.summary,
            "labels": list(self.labels),
            "backend": backend,
            "model_path": model_path,
            "model_available": os.path.exists(model_path),
            "local_training_model_path": self.model_path,
            "tflite_model_path": self.tflite_model_path,
            "inference_profile_path": self.inference_profile_path,
            "inference_profile_available": os.path.exists(self.inference_profile_path),
        }


def _env_or_default(env_name: str, default_value: str) -> str:
    return os.getenv(env_name, default_value)


def _modality_registry() -> dict[str, ModalityConfig]:
    chest_ct = ModalityConfig(
        key="chest_ct",
        display_name="Chest CT Scan",
        upload_title="Upload a chest CT image",
        upload_hint="PNG, JPG, or JPEG chest CT slices for adenocarcinoma vs normal classification.",
        summary="Deep-learning chest CT cancer classification using the existing VGG16-based pipeline.",
        labels=("Adenocarcinoma Cancer", "Normal"),
        preprocessing="vgg16",
        model_path=_env_or_default("MODEL_PATH", os.path.join("model", "model.h5")),
        tflite_model_path=_env_or_default("TFLITE_MODEL_PATH", os.path.join("model", "model.tflite")),
        inference_profile_path=_env_or_default(
            "INFERENCE_PROFILE_PATH", os.path.join("model", "inference_profile.json")
        ),
        unsupported_image_message="The uploaded image does not match the chest CT scans the model was trained on.",
    )
    ecg = ModalityConfig(
        key="ecg",
        display_name="ECG Image",
        upload_title="Upload an ECG image",
        upload_hint="PNG, JPG, or JPEG ECG printouts for cardiac condition classification.",
        summary="Transfer-learning ECG image classifier for normal traces, abnormal heartbeat, and history of MI.",
        labels=("Abnormal heartbeat", "History of MI", "Normal Person"),
        preprocessing="mobilenet_v2",
        model_path=_env_or_default("ECG_MODEL_PATH", os.path.join("artifacts", "ecg_training", "model.h5")),
        tflite_model_path=_env_or_default(
            "ECG_TFLITE_MODEL_PATH", os.path.join("model", "ecg_model.tflite")
        ),
        inference_profile_path=_env_or_default(
            "ECG_INFERENCE_PROFILE_PATH", os.path.join("model", "ecg_inference_profile.json")
        ),
        unsupported_image_message="The uploaded image does not match the ECG images the model was trained on.",
    )
    return {
        chest_ct.key: chest_ct,
        ecg.key: ecg,
    }


def get_default_modality_key() -> str:
    return "chest_ct"


def list_modality_configs() -> list[ModalityConfig]:
    return list(_modality_registry().values())


def get_modality_config(modality: str | None) -> ModalityConfig:
    normalized = (modality or get_default_modality_key()).strip().lower()
    aliases = {
        "ct": "chest_ct",
        "chest": "chest_ct",
        "chest_ct": "chest_ct",
        "ecg": "ecg",
    }
    modality_key = aliases.get(normalized)
    if modality_key is None:
        raise ValueError(f"Unsupported modality: {modality}")
    return _modality_registry()[modality_key]
