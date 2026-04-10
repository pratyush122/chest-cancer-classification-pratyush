import os
from pathlib import Path

from dotenv import load_dotenv

from cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    EvaluationConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
)
from cnnClassifier.utils.common import create_directories, read_yaml

load_dotenv()


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        data_ingestion_config_values = self.config.data_ingestion
        create_directories([data_ingestion_config_values.root_dir])

        source_url = os.getenv("DATA_SOURCE_URL", data_ingestion_config_values.source_URL)
        return DataIngestionConfig(
            root_dir=Path(data_ingestion_config_values.root_dir),
            source_URL=source_url,
            local_data_file=Path(data_ingestion_config_values.local_data_file),
            unzip_dir=Path(data_ingestion_config_values.unzip_dir),
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        base_model_config_values = self.config.prepare_base_model
        create_directories([base_model_config_values.root_dir])

        return PrepareBaseModelConfig(
            root_dir=Path(base_model_config_values.root_dir),
            base_model_path=Path(base_model_config_values.base_model_path),
            updated_base_model_path=Path(base_model_config_values.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
            params_random_seed=self.params.RANDOM_SEED,
        )

    def get_training_config(self) -> TrainingConfig:
        training_config_values = self.config.training
        base_model_config_values = self.config.prepare_base_model
        training_data_path = Path(self.config.data_ingestion.unzip_dir) / "Chest-CT-Scan-data"

        create_directories([Path(training_config_values.root_dir)])

        return TrainingConfig(
            root_dir=Path(training_config_values.root_dir),
            trained_model_path=Path(training_config_values.trained_model_path),
            updated_base_model_path=Path(base_model_config_values.updated_base_model_path),
            training_data=training_data_path,
            params_epochs=self.params.EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
            params_is_augmentation=self.params.AUGMENTATION,
            params_image_size=self.params.IMAGE_SIZE,
            params_random_seed=self.params.RANDOM_SEED,
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        trained_model_path = os.getenv("TRAINED_MODEL_PATH", "artifacts/training/model.h5")
        training_data_path = os.getenv(
            "TRAINING_DATA_PATH", "artifacts/data_ingestion/Chest-CT-Scan-data"
        )
        mlflow_tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            "https://dagshub.com/entbappy/chest-Disease-Classification-MLflow-DVC.mlflow",
        )

        return EvaluationConfig(
            path_of_model=Path(trained_model_path),
            training_data=Path(training_data_path),
            mlflow_uri=mlflow_tracking_uri,
            all_params=dict(self.params),
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
        )
