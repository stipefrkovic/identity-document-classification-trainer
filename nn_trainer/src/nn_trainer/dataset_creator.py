from abc import ABC, abstractmethod

import tensorflow as tf
import os


class DatasetCreator(ABC):
    @abstractmethod
    def create_dataset(self) -> dict:
        pass


class KerasDatasetCreator(DatasetCreator):
    def __init__(self, path=os.getcwd() + '/nn_trainer/dataset', image_size=(224, 224), batch_size=1) -> None:
        self.dataset_path = path
        self.image_size = image_size
        self.batch_size = batch_size

    def create_dataset(self) -> dict:
        """Creates the dataset from the folder at the provided path.

    Returns:
        dict: {
          dataset: Dataset
        }
    """
        dataset = tf.keras.utils.image_dataset_from_directory(self.dataset_path,
                                                              shuffle=True,
                                                              batch_size=self.batch_size,
                                                              image_size=self.image_size)
        train_dataset, test_dataset = tf.keras.utils.split_dataset(dataset,
                                                                   left_size=0.8)
        reduced_train_dataset, validation_dataset = tf.keras.utils.split_dataset(train_dataset,
                                                                                 left_size=0.8)
        print(int(reduced_train_dataset.cardinality()))
        print(int(validation_dataset.cardinality()))
        print(int(test_dataset.cardinality()))

        return {
            "dataset": dataset
        }
