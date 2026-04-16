import math
import tensorflow as tf
from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path

from cnnClassifier.utils.data_utils import build_grouped_split_dataframe



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        tf.keras.utils.set_random_seed(self.config.params_random_seed)

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        preprocessing_function = tf.keras.applications.vgg16.preprocess_input

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="categorical",
            x_col="filepath",
            y_col="label",
            seed=self.config.params_random_seed,
        )

        train_dataframe, validation_dataframe, split_summary = build_grouped_split_dataframe(
            dataset_dir=self.config.training_data,
            validation_split=self.config.params_validation_split,
            seed=self.config.params_random_seed,
        )
        self.split_summary = split_summary

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocessing_function
        )

        self.valid_generator = valid_datagenerator.flow_from_dataframe(
            dataframe=validation_dataframe,
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=preprocessing_function,
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
            )
        else:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=preprocessing_function
            )

        self.train_generator = train_datagenerator.flow_from_dataframe(
            dataframe=train_dataframe,
            shuffle=True,
            **dataflow_kwargs
        )
        class_counts = train_dataframe["label"].value_counts().to_dict()
        total_samples = float(len(train_dataframe))
        class_name_to_index = self.train_generator.class_indices
        self.class_weight = {
            class_name_to_index[label]: total_samples / (len(class_counts) * count)
            for label, count in class_counts.items()
        }

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)



    
    def train(self):
        self.steps_per_epoch = max(1, math.ceil(self.train_generator.samples / self.train_generator.batch_size))
        self.validation_steps = max(1, math.ceil(self.valid_generator.samples / self.valid_generator.batch_size))

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            class_weight=self.class_weight,
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
