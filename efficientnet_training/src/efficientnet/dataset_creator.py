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
    def create_dataset(self, dataset_path, image_size=224, batch_size=16):
        full_dataset_path = str(Path().absolute()) + dataset_path
        logger.debug("Full dataset path: " + str(full_dataset_path))

        logger.debug("Loading dataset")
        dataset = tf.keras.utils.image_dataset_from_directory(full_dataset_path,
                                                              shuffle=True,
                                                              batch_size=batch_size,
                                                              image_size=(image_size, image_size),
                                                              label_mode="categorical")

        num_classes=len(dataset.class_names)
        logger.debug("Num of classes: " + str(num_classes))

        return {
            "dataset": dataset,
            "num_classes": num_classes
        }

    def split_dataset(self, dataset, train_split=0.7, validation_split=0.15, test_split=0.15):
        if train_split + validation_split + test_split != 1.0:
            raise ValueError("The dataset splits do not add up to 1.")

        dataset_size = dataset.cardinality().numpy()
        logger.info("Size of full dataset: " + str(dataset_size))

        train_dataset_size = int(train_split * dataset_size)
        validation_dataset_size = int(validation_split * dataset_size)
        test_dataset_size = int(test_split * dataset_size)
        logger.debug("Desired size of train dataset: " + str(train_dataset_size))
        logger.debug("Desired size of validation dataset: " + str(validation_dataset_size))
        logger.debug("Desired size of test dataset: " + str(test_dataset_size))

        train_dataset = dataset.take(train_dataset_size)
        validation_dataset = dataset.skip(train_dataset_size).take(validation_dataset_size)
        test_dataset = dataset.skip(train_dataset_size).skip(validation_dataset_size)

        logger.info("Size of train dataset: " + str(train_dataset.cardinality().numpy()))
        logger.info("Size of test dataset: " + str(validation_dataset.cardinality().numpy()))
        logger.info("Size of validation dataset: " + str(test_dataset.cardinality().numpy()))

        train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return {
            "train_dataset": train_dataset,
            "validation_dataset": validation_dataset,
            "test_dataset": test_dataset
        }
