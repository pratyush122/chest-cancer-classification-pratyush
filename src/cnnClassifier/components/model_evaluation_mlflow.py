import os
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import mlflow.keras
import tensorflow as tf

from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        data_generator_options = dict(rescale=1.0 / 255, validation_split=0.30)
        data_flow_options = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
        )

        validation_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            **data_generator_options
        )

        self.valid_generator = validation_data_generator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **data_flow_options,
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": float(self.score[0]), "accuracy": float(self.score[1])}
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
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})

            if tracking_uri_scheme != "file":
                mlflow.keras.log_model(
                    self.model,
                    artifact_path="model",
                    registered_model_name="VGG16Model",
                )
            else:
                mlflow.keras.log_model(self.model, artifact_path="model")
