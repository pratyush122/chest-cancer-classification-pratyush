import hashlib
import json
import random
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf

from cnnClassifier.utils.common import create_directories, save_json
from cnnClassifier.utils.image_features import build_ood_profile, extract_image_statistics


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "datasets" / "ecg"
DATASET_ZIP_PATH = DATASET_DIR / "ecg_heart_conditions.zip"
DATASET_EXTRACT_DIR = DATASET_DIR / "raw"
DATASET_IMAGE_DIR = DATASET_EXTRACT_DIR / "ECG Dataset"
DATASET_DOWNLOAD_URL = "https://zenodo.org/api/records/14767442/files/archive%20(10).zip/content"
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "ecg_training"
MODEL_DIR = ROOT_DIR / "model"
SCORES_PATH = ROOT_DIR / "ecg_scores.json"
MODEL_PATH = ARTIFACT_DIR / "model.h5"
BEST_CHECKPOINT_PATH = ARTIFACT_DIR / "best_model.keras"
HISTORY_PATH = ARTIFACT_DIR / "history.json"
TFLITE_PATH = MODEL_DIR / "ecg_model.tflite"
INFERENCE_PROFILE_PATH = MODEL_DIR / "ecg_inference_profile.json"
METADATA_PATH = MODEL_DIR / "ecg_metadata.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2
SEED = 42
INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 4
LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5
NORMAL_PERSON_TARGET_RATIO = 0.8


def ensure_dataset() -> Path:
    create_directories([DATASET_DIR, DATASET_EXTRACT_DIR])
    if DATASET_IMAGE_DIR.exists():
        return DATASET_IMAGE_DIR

    if not DATASET_ZIP_PATH.exists():
        urllib.request.urlretrieve(DATASET_DOWNLOAD_URL, DATASET_ZIP_PATH)

    with zipfile.ZipFile(DATASET_ZIP_PATH, "r") as zip_file:
        zip_file.extractall(DATASET_EXTRACT_DIR)

    return DATASET_IMAGE_DIR


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probabilities: np.ndarray,
    class_names: list[str],
) -> dict:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    y_probabilities = np.asarray(y_probabilities, dtype=np.float32)
    confusion_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
    for actual_label, predicted_label in zip(y_true, y_pred):
        confusion_matrix[int(actual_label), int(predicted_label)] += 1

    per_class_metrics = {}
    class_precisions = []
    class_recalls = []
    class_f1_scores = []

    for class_index, class_name in enumerate(class_names):
        true_positive = confusion_matrix[class_index, class_index]
        false_positive = confusion_matrix[:, class_index].sum() - true_positive
        false_negative = confusion_matrix[class_index, :].sum() - true_positive
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        class_precisions.append(float(precision))
        class_recalls.append(float(recall))
        class_f1_scores.append(float(f1_score))
        per_class_metrics[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1_score),
            "support": int(confusion_matrix[class_index, :].sum()),
        }

    return {
        "accuracy": float((y_true == y_pred).mean()),
        "precision_macro": float(np.mean(class_precisions)),
        "recall_macro": float(np.mean(class_recalls)),
        "f1_macro": float(np.mean(class_f1_scores)),
        "confusion_matrix": confusion_matrix.tolist(),
        "per_class": per_class_metrics,
    }


def _file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_ecg_split_dataframe(dataset_dir: Path):
    train_rows = []
    validation_rows = []
    duplicate_summary = {}

    for class_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        grouped_by_hash = defaultdict(list)
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            grouped_by_hash[_file_md5(image_path)].append(image_path)

        unique_groups = list(grouped_by_hash.values())
        rng = random.Random(SEED)
        rng.shuffle(unique_groups)

        if len(unique_groups) == 1:
            validation_group_count = 1
        else:
            validation_group_count = max(1, round(len(unique_groups) * VALIDATION_SPLIT))
            validation_group_count = min(validation_group_count, len(unique_groups) - 1)

        validation_groups = unique_groups[:validation_group_count]
        training_groups = unique_groups[validation_group_count:]

        for group in training_groups:
            # Keep one canonical image per duplicate group so the model does not
            # overfit repeated ECG renderings of the same trace.
            train_rows.append({"filepath": str(group[0]), "label": class_dir.name})

        for group in validation_groups:
            validation_rows.append({"filepath": str(group[0]), "label": class_dir.name})

        total_images = sum(len(group) for group in unique_groups)
        duplicate_summary[class_dir.name] = {
            "total_images": total_images,
            "unique_images": len(unique_groups),
            "duplicates_removed": total_images - len(unique_groups),
            "training_images": len(training_groups),
            "training_unique_images": len(training_groups),
            "validation_unique_images": len(validation_groups),
        }

    import pandas as pd

    return pd.DataFrame(train_rows), pd.DataFrame(validation_rows), duplicate_summary


def build_balanced_training_dataframe(train_df):
    class_counts = train_df["label"].value_counts().to_dict()
    max_class_count = max(class_counts.values())
    median_class_count = int(round(float(np.median(list(class_counts.values())))))
    normal_target_count = int(round(max_class_count * NORMAL_PERSON_TARGET_RATIO))

    target_counts = {
        label: max(count, median_class_count)
        for label, count in class_counts.items()
    }
    if "Normal Person" in target_counts:
        target_counts["Normal Person"] = max(
            target_counts["Normal Person"],
            normal_target_count,
        )

    balanced_train_df = (
        train_df.groupby("label", group_keys=False)
        .apply(
            lambda group: group.sample(
                n=target_counts[group.name],
                replace=len(group) < target_counts[group.name],
                random_state=SEED,
            )
        )
        .reset_index(drop=True)
    )
    return balanced_train_df, target_counts


def create_generators(dataset_dir: Path):
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    train_df, validation_df, split_summary = build_ecg_split_dataframe(dataset_dir)
    balanced_train_df, target_counts = build_balanced_training_dataframe(train_df)

    dataflow_kwargs = dict(
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        interpolation="bilinear",
        class_mode="categorical",
        x_col="filepath",
        y_col="label",
        seed=SEED,
    )
    train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.08,
    )
    validation_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagenerator.flow_from_dataframe(
        dataframe=balanced_train_df,
        shuffle=True,
        **dataflow_kwargs,
    )
    validation_generator = validation_datagenerator.flow_from_dataframe(
        dataframe=validation_df,
        shuffle=False,
        **dataflow_kwargs,
    )

    class_counts = target_counts
    total_samples = float(len(balanced_train_df))
    class_name_to_index = train_generator.class_indices
    class_weight = {
        class_name_to_index[label]: total_samples / (len(class_counts) * count)
        for label, count in class_counts.items()
    }

    return train_df, validation_df, split_summary, train_generator, validation_generator, class_weight


def build_model(num_classes: int):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def train_model(train_generator, validation_generator, class_weight):
    model, base_model = build_model(len(train_generator.class_indices))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            BEST_CHECKPOINT_PATH,
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history_initial = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=INITIAL_EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_finetune = model.fit(
        train_generator,
        validation_data=validation_generator,
        initial_epoch=len(history_initial.history["loss"]),
        epochs=len(history_initial.history["loss"]) + FINE_TUNE_EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    if BEST_CHECKPOINT_PATH.exists():
        model = tf.keras.models.load_model(BEST_CHECKPOINT_PATH)

    combined_history = {}
    for history in (history_initial.history, history_finetune.history):
        for key, values in history.items():
            combined_history.setdefault(key, []).extend([float(value) for value in values])
    return model, combined_history


def evaluate_model(model, validation_generator) -> dict:
    scores = model.evaluate(validation_generator, verbose=0)
    probabilities = model.predict(validation_generator, verbose=0)
    predicted_labels = probabilities.argmax(axis=1)
    actual_labels = validation_generator.classes
    class_names = [
        class_name
        for class_name, _ in sorted(validation_generator.class_indices.items(), key=lambda item: item[1])
    ]
    metrics = calculate_classification_metrics(
        y_true=actual_labels,
        y_pred=predicted_labels,
        y_probabilities=probabilities,
        class_names=class_names,
    )
    metrics["loss"] = float(scores[0])
    return metrics


def save_ecg_inference_profile(train_df):
    feature_dicts = [
        extract_image_statistics(image_path)
        for image_path in train_df["filepath"].tolist()
    ]
    inference_profile = {
        "author": "Pratyush Mishra",
        "model_type": "mobilenetv2_ecg_image_classifier",
        "source_model_path": str(MODEL_PATH.relative_to(ROOT_DIR)),
        "ood_profile": build_ood_profile(feature_dicts),
    }
    save_json(INFERENCE_PROFILE_PATH, inference_profile)


def export_tflite_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    TFLITE_PATH.write_bytes(tflite_model)


def count_dataset_images(dataset_dir: Path) -> dict[str, int]:
    return {
        class_dir.name: len(list(class_dir.glob("*")))
        for class_dir in sorted(dataset_dir.iterdir())
        if class_dir.is_dir()
    }


def main():
    tf.keras.utils.set_random_seed(SEED)
    create_directories([ARTIFACT_DIR, MODEL_DIR])
    dataset_dir = ensure_dataset()
    train_df, validation_df, split_summary, train_generator, validation_generator, class_weight = create_generators(
        dataset_dir
    )
    model, history = train_model(train_generator, validation_generator, class_weight)
    model.save(MODEL_PATH)
    metrics = evaluate_model(model, validation_generator)
    save_ecg_inference_profile(train_df)
    export_tflite_model(model)

    class_names = [
        class_name
        for class_name, _ in sorted(train_generator.class_indices.items(), key=lambda item: item[1])
    ]
    score_payload = {
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "confusion_matrix": metrics["confusion_matrix"],
        "per_class": metrics["per_class"],
        "validation_samples": int(len(validation_generator.filenames)),
        "duplicate_summary": split_summary,
        "class_names": class_names,
    }
    save_json(SCORES_PATH, score_payload)
    save_json(HISTORY_PATH, history)

    metadata = {
        "author": "Pratyush Mishra",
        "dataset_name": "ECG Dataset for Heart Condition Classification",
        "dataset_download_url": DATASET_DOWNLOAD_URL,
        "classes": class_names,
        "dataset_counts": count_dataset_images(dataset_dir),
        "validation_summary": split_summary,
        "metrics": score_payload,
        "artifacts": {
            "trained_model_path": str(MODEL_PATH.relative_to(ROOT_DIR)),
            "tflite_model_path": str(TFLITE_PATH.relative_to(ROOT_DIR)),
            "inference_profile_path": str(INFERENCE_PROFILE_PATH.relative_to(ROOT_DIR)),
        },
    }
    save_json(METADATA_PATH, metadata)

    if BEST_CHECKPOINT_PATH.exists():
        BEST_CHECKPOINT_PATH.unlink()

    print(json.dumps(metadata["metrics"], indent=2))


if __name__ == "__main__":
    main()
