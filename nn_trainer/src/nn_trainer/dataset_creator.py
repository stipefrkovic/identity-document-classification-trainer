from abc import ABC, abstractmethod

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np


class DatasetCreator(ABC):
    @abstractmethod
    def create_dataset(self) -> dict:
        pass


class KerasEfficientNetDatasetCreator(DatasetCreator):
    def __init__(self, path=os.getcwd() + '/nn_trainer/dataset', image_size=(224, 224), batch_size=1) -> None:
        self.dataset_path = path
        self.image_size = image_size
        self.image_shape = self.image_size + (3,)
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
                                                              image_size=self.image_size,
                                                              label_mode="categorical")

        # for image_batch, labels_batch in dataset:
        #     print(image_batch.shape)
        #     print(labels_batch.shape)

        class_names = dataset.class_names
        for i, (image, label) in enumerate(dataset.take(9)):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image[0].numpy().astype("uint8"))
            # plt.title("{}".format(class_names(label)))
            plt.axis("off")
        plt.show()

        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomRotation(factor=0.15),
                tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
                tf.keras.layers.RandomFlip(),
                tf.keras.layers.RandomContrast(factor=0.1),
            ],
            name="data_augmentation",
        )

        for i, (image, label) in enumerate(dataset.take(9)):
            ax = plt.subplot(3, 3, i + 1)
            aug_img = data_augmentation(tf.expand_dims(image[0], axis=0))
            plt.imshow(aug_img[0].numpy().astype("uint8"))
            plt.axis("off")
        plt.show()

        # train_dataset, test_dataset = tf.keras.utils.split_dataset(dataset,
        #                                                            left_size=0.8)
        # reduced_train_dataset, validation_dataset = tf.keras.utils.split_dataset(train_dataset,
        #                                                                          left_size=0.8)
        # print(int(reduced_train_dataset.cardinality()))
        # print(int(validation_dataset.cardinality()))
        # print(int(test_dataset.cardinality()))

        # train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        # validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        # test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return {
            "dataset": dataset
        }
