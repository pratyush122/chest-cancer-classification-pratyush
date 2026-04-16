import os
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf

from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.data_utils import build_grouped_split_dataframe
from cnnClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        data_flow_options = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="categorical",
            x_col="filepath",
            y_col="label",
        )
        _, validation_dataframe, split_summary = build_grouped_split_dataframe(
            dataset_dir=self.config.training_data,
            validation_split=self.config.params_validation_split,
            seed=self.config.params_random_seed,
        )
        self.split_summary = split_summary

        validation_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.vgg16.preprocess_input
        )

        self.valid_generator = validation_data_generator.flow_from_dataframe(
            dataframe=validation_dataframe,
            shuffle=False,
            **data_flow_options,
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    @staticmethod
    def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        positive_mask = y_true == 1
        negative_mask = y_true == 0
        positive_count = int(positive_mask.sum())
        negative_count = int(negative_mask.sum())
        if positive_count == 0 or negative_count == 0:
            return 0.0

        order = np.argsort(y_score)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
        positive_rank_sum = ranks[positive_mask].sum()
        auc = (
            positive_rank_sum
            - (positive_count * (positive_count + 1) / 2.0)
        ) / (positive_count * negative_count)
        return float(auc)

    def _classification_metrics(
        self,
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
        class_f1_scores = []
        class_precisions = []
        class_recalls = []
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

        positive_class_index = class_names.index("normal") if "normal" in class_names else 1
        auc = self._binary_auc(
            (y_true == positive_class_index).astype(np.int32),
            y_probabilities[:, positive_class_index],
        )

        return {
            "accuracy": float((y_true == y_pred).mean()),
            "precision_macro": float(np.mean(class_precisions)),
            "recall_macro": float(np.mean(class_recalls)),
            "f1_macro": float(np.mean(class_f1_scores)),
            "roc_auc": auc,
            "confusion_matrix": confusion_matrix.tolist(),
            "per_class": per_class_metrics,
        }

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator, verbose=0)
        probabilities = self.model.predict(self.valid_generator, verbose=0)
        predicted_labels = probabilities.argmax(axis=1)
        actual_labels = self.valid_generator.classes
        class_names = [
            class_name
            for class_name, _ in sorted(self.valid_generator.class_indices.items(), key=lambda item: item[1])
        ]
        self.metric_summary = self._classification_metrics(
            y_true=actual_labels,
            y_pred=predicted_labels,
            y_probabilities=probabilities,
            class_names=class_names,
        )
        self.save_score()

    def save_score(self):
        scores = {
            "loss": float(self.score[0]),
            "accuracy": float(self.metric_summary["accuracy"]),
            "precision_macro": float(self.metric_summary["precision_macro"]),
            "recall_macro": float(self.metric_summary["recall_macro"]),
            "f1_macro": float(self.metric_summary["f1_macro"]),
            "roc_auc": float(self.metric_summary["roc_auc"]),
            "confusion_matrix": self.metric_summary["confusion_matrix"],
            "per_class": self.metric_summary["per_class"],
            "validation_samples": int(len(self.valid_generator.filenames)),
            "duplicate_summary": self.split_summary,
        }
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
        mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")

        if mlflow_username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
        if mlflow_password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password

        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_uri_scheme = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(dict(self.config.all_params))
            mlflow.log_metrics(
                {
                    "loss": float(self.score[0]),
                    "accuracy": float(self.metric_summary["accuracy"]),
                    "precision_macro": float(self.metric_summary["precision_macro"]),
                    "recall_macro": float(self.metric_summary["recall_macro"]),
                    "f1_macro": float(self.metric_summary["f1_macro"]),
                    "roc_auc": float(self.metric_summary["roc_auc"]),
                }
            )

            if tracking_uri_scheme != "file":
                mlflow.keras.log_model(
                    self.model,
                    artifact_path="model",
                    registered_model_name="VGG16Model",
                )
            else:
                mlflow.keras.log_model(self.model, artifact_path="model")
