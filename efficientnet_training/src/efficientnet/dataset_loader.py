from abc import ABC, abstractmethod
from tensorflow.keras import utils
from tensorflow import data
from pathlib import Path

from utils.logger import logger


class DatasetLoader(ABC):
    def __init__(self):
        self.dataset = None
        self.num_classes = None
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

    def get_num_classes(self):
        if self.num_classes is None:
            logger.error("Dataset not loaded.")
            exit(1)
        else:
            return self.num_classes

    def get_train_dataset(self):
        if self.train_dataset is None:
            logger.error("Dataset not split.")
            exit(1)
        else:
            return self.train_dataset

    def get_validation_dataset(self):
        if self.validation_dataset is None:
            logger.error("Dataset not split.")
            exit(1)
        else:
            return self.validation_dataset

    def get_test_dataset(self):
        if self.test_dataset is None:
            logger.error("Dataset not split.")
            exit(1)
        else:
            return self.test_dataset

    @abstractmethod
    def load_dataset(self, dataset_path, image_size, batch_size):
        pass

    @abstractmethod
    def split_dataset(self, train_split, validation_split, test_split):
        pass


class KerasImageDatasetLoader(DatasetLoader):
    def __init__(self):
        super().__init__()

    def load_dataset(self, dataset_path, image_size, batch_size=8):
        logger.debug(f"Image size: {image_size}")
        logger.info(f"Batch size: {batch_size}")

        full_dataset_path = str(Path().absolute()) + dataset_path
        logger.debug(f"Full dataset path: {full_dataset_path}")
        logger.debug("Loading dataset")
        dataset = utils.image_dataset_from_directory(full_dataset_path,
                                                     shuffle=True,
                                                     batch_size=batch_size,
                                                     image_size=(image_size, image_size),
                                                     label_mode="categorical")
        logger.info("Loaded dataset")
        self.dataset = dataset

        num_classes = len(dataset.class_names)
        logger.info(f"Num of classes: {num_classes}")
        self.num_classes = num_classes

    def split_dataset(self, train_split, validation_split, test_split):
        if train_split + validation_split + test_split != 1.0:
            logger.error("The dataset splits do not add up to 1.")
            exit(1)

        dataset_size = self.dataset.cardinality().numpy()
        logger.info(f"Size of full dataset: {dataset_size}")

        train_dataset_size = int(train_split * dataset_size)
        validation_dataset_size = int(validation_split * dataset_size)
        test_dataset_size = int(test_split * dataset_size)
        if train_dataset_size == 0 or validation_dataset_size == 0 or test_split == 0:
            logger.error(
                "Desired size of a dataset is 0: modify the dataset split ratios/batch size or use a bigger dataset.")
            exit(1)
        logger.debug(f"Desired size of train dataset: {train_dataset_size}")
        logger.debug(f"Desired size of validation dataset: {validation_dataset_size}")
        logger.debug(f"Desired size of test dataset: {test_dataset_size}")

        train_dataset = self.dataset.take(train_dataset_size)
        validation_dataset = self.dataset.skip(train_dataset_size).take(validation_dataset_size)
        test_dataset = self.dataset.skip(train_dataset_size).skip(validation_dataset_size)
        logger.info(f"Size of train dataset: {train_dataset.cardinality().numpy()}")
        logger.info(f"Size of test dataset: {validation_dataset.cardinality().numpy()}")
        logger.info(f"Size of validation dataset: {test_dataset.cardinality().numpy()}")

        train_dataset = train_dataset.prefetch(buffer_size=data.AUTOTUNE)
        validation_dataset = validation_dataset.prefetch(buffer_size=data.AUTOTUNE)
        test_dataset = test_dataset.prefetch(buffer_size=data.AUTOTUNE)

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
