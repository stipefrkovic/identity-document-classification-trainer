from abc import ABC, abstractmethod
import tensorflow as tf
from pathlib import Path
from logger import logger

class DatasetCreator(ABC):
    @abstractmethod
    def create_dataset(self):
        pass

    @abstractmethod
    def split_dataset(self, dataset, train_split=0.7, validation_split=0.15, test_split=0.15):
        pass


class KerasEfficientNetDatasetCreator(DatasetCreator):
    def create_dataset(self, dataset_path, image_size=224, batch_size=8):
        logger.debug(f"Image size: {image_size}")
        logger.info(f"Batch size: {batch_size}")


        full_dataset_path = str(Path().absolute()) + dataset_path
        logger.debug(f"Full dataset path: {full_dataset_path}")

        logger.debug("Loading dataset")
        dataset = tf.keras.utils.image_dataset_from_directory(full_dataset_path,
                                                                shuffle=True,
                                                                batch_size=batch_size,
                                                                image_size=(image_size, image_size),
                                                                label_mode="categorical")

        num_classes=len(dataset.class_names)
        logger.info(f"Num of classes: {num_classes}")

        return {
            "dataset": dataset,
            "num_classes": num_classes
        }   

    def split_dataset(self, dataset, train_split=0.7, validation_split=0.15, test_split=0.15):
        if train_split + validation_split + test_split != 1.0:
            raise ValueError("The dataset splits do not add up to 1.")

        dataset_size = dataset.cardinality().numpy()
        logger.info(f"Size of full dataset: {dataset_size}")

        train_dataset_size = int(train_split * dataset_size)
        validation_dataset_size = int(validation_split * dataset_size)
        test_dataset_size = int(test_split * dataset_size)
        if train_dataset_size == 0 or validation_dataset_size == 0 or test_split == 0:
            raise ValueError("Desired size of one of the datasets is 0: please modify the dataset split ratios or increase the size of the dataset")
        logger.debug(f"Desired size of train dataset: {train_dataset_size}")
        logger.debug(f"Desired size of validation dataset: {validation_dataset_size}")
        logger.debug(f"Desired size of test dataset: {test_dataset_size}")

        train_dataset = dataset.take(train_dataset_size)
        validation_dataset = dataset.skip(train_dataset_size).take(validation_dataset_size)
        test_dataset = dataset.skip(train_dataset_size).skip(validation_dataset_size)

        logger.info(f"Size of train dataset: {train_dataset.cardinality().numpy()}")
        logger.info(f"Size of test dataset: {validation_dataset.cardinality().numpy()}")
        logger.info(f"Size of validation dataset: {test_dataset.cardinality().numpy()}")

        train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return {
            "train_dataset": train_dataset,
            "validation_dataset": validation_dataset,
            "test_dataset": test_dataset
        }
