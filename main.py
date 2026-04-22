import runpy
from pathlib import Path
from typing import Callable

from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import (
    PrepareBaseModelTrainingPipeline,
)
from cnnClassifier.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline

ROOT_DIR = Path(__file__).resolve().parent


def run_pipeline_stage(stage_name: str, stage_runner: Callable[[], None]) -> None:
    try:
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")
        stage_runner()
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
    except Exception as error:
        logger.exception(error)
        raise error


if __name__ == "__main__":
    run_pipeline_stage("Data Ingestion stage", lambda: DataIngestionTrainingPipeline().main())
    run_pipeline_stage("Prepare base model", lambda: PrepareBaseModelTrainingPipeline().main())
    run_pipeline_stage("Training", lambda: ModelTrainingPipeline().main())
    run_pipeline_stage("Evaluation stage", lambda: EvaluationPipeline().main())
    run_pipeline_stage(
        "ECG model training",
        lambda: runpy.run_path(str(ROOT_DIR / "scripts" / "train_ecg_model.py"), run_name="__main__"),
    )
