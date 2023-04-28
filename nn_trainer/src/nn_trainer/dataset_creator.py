from abc import ABC, abstractmethod

import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt


class DatasetCreator(ABC):
    def __init__(self):
        self.dataset = None
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

    @abstractmethod
    def create_dataset(self):
        pass


class KerasEfficientNetDatasetCreator(DatasetCreator):
    def __init__(self, dataset_path='/nn_trainer/dataset', image_size=(224, 224), batch_size=1):
        super().__init__()
        self.dataset_path = str(Path().absolute()) + dataset_path
        self.image_size = image_size
        self.image_shape = self.image_size + (3,)
        self.batch_size = batch_size

    def create_dataset(self):
        dataset = tf.keras.utils.image_dataset_from_directory(self.dataset_path,
                                                              shuffle=True,
                                                              batch_size=self.batch_size,
                                                              image_size=self.image_size,
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

        self.dataset = dataset

        return dataset

    def split_dataset(self, test_train_ratio=0.8, train_validation_ratio=0.8):
        train_dataset, test_dataset = tf.keras.utils.split_dataset(self.dataset,
                                                                   left_size=test_train_ratio)
        reduced_train_dataset, validation_dataset = tf.keras.utils.split_dataset(train_dataset,
                                                                                 left_size=train_validation_ratio)
        print(int(reduced_train_dataset.cardinality()))
        print(int(validation_dataset.cardinality()))
        print(int(test_dataset.cardinality()))

        self.train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
