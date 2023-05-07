from abc import ABC, abstractmethod

import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

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
        dataset = tf.keras.utils.image_dataset_from_directory(full_dataset_path,
                                                              shuffle=True,
                                                              batch_size=batch_size,
                                                              image_size=(image_size, image_size),
                                                              label_mode="categorical")

        # for image_batch, labels_batch in dataset:
        #     print(image_batch.shape)
        #     print(labels_batch.shape)

        # show a few images from the dataset
        class_names = dataset.class_names
        for i, (image, label) in enumerate(dataset.take(9)):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image[0].numpy().astype("uint8"))
            # plt.title("{}".format(class_names(label)))
            plt.axis("off")
        plt.show()

        # augment and show a few images from the dataset
        # data_augmentation = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.RandomRotation(factor=0.1),
        #         tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
        #         tf.keras.layers.RandomFlip(),
        #         tf.keras.layers.RandomContrast(factor=0.1),
        #         tf.keras.layers.RandomZoom(height_factor=(-0.1, 0), width_factor=(-0.1, 0))
        #     ],
        #     name="data_augmentation",
        # )
        # for i, (image, label) in enumerate(dataset.take(9)):
        #     ax = plt.subplot(3, 3, i + 1)
        #     aug_img = data_augmentation(tf.expand_dims(image[0], axis=0))
        #     plt.imshow(aug_img[0].numpy().astype("uint8"))
        #     plt.axis("off")
        # plt.show()

        num_classes=len(dataset.class_names)
        logger.info("num_classes: " + str(num_classes))

        return {
            "dataset": dataset,
            "num_classes": num_classes
        }

    def split_dataset(self, dataset, train_split=0.7, validation_split=0.15, test_split=0.15):
        if train_split + validation_split + test_split != 1.0:
            raise Exception("The dataset splits do not add up to 1.")

        dataset_size = dataset.cardinality().numpy()
        logger.info("dataset_size: " + str(dataset_size))

        train_dataset_size = int(train_split * dataset_size)
        logger.info("train_dataset_size: " + str(train_dataset_size))
        validation_dataset_size = int(validation_split * dataset_size)
        logger.info("validation_dataset_size: " + str(validation_dataset_size))
        test_dataset_size = int(test_split * dataset_size)
        logger.info("test_dataset_size: " + str(test_dataset_size))

        train_dataset = dataset.take(train_dataset_size)
        validation_dataset = dataset.skip(train_dataset_size).take(validation_dataset_size)
        test_dataset = dataset.skip(train_dataset_size).skip(validation_dataset_size)

        logger.info("train_dataset.cardinality(): " + str(train_dataset.cardinality()))
        logger.info("validation_dataset.cardinality(): " + str(validation_dataset.cardinality()))
        logger.info("test_dataset.cardinality(): " + str(test_dataset.cardinality()))

        train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return {
            "train_dataset": train_dataset,
            "validation_dataset": validation_dataset,
            "test_dataset": test_dataset
        }
