from abc import ABC, abstractmethod
from tensorflow.keras import utils
from tensorflow import data
from pathlib import Path

from utils.logger import logger


class DatasetLoader(ABC):
    """
    ABC for a Dataset Loader.
    """
    def __init__(self, dataset_path):
        """
        Initializes the Dataset Loader.
        @param dataset_path: Path to directory containing the dataset.
        """
        self.dataset_path = dataset_path
        self.dataset = None
        self.num_classes = None
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

    def get_num_classes(self):
        """
        Gets the number of classes in the dataset.
        @return: number of classes in the dataset.
        """
        if self.num_classes is not None:
            return self.num_classes
        logger.error("Dataset not loaded")
        exit(1)

    def get_train_dataset(self):
        """
        Gets the train dataset.
        @return: train dataset.
        """
        if self.train_dataset is not None:
            return self.train_dataset
        logger.error("Dataset not split")
        exit(1)

    def get_validation_dataset(self):
        """
        Gets the validation dataset.
        @return: validation dataset.
        """
        if self.validation_dataset is not None:
            return self.validation_dataset
        logger.error("Dataset not split")
        exit(1)

    def get_test_dataset(self):
        """
        Gets the test dataset.
        @return: test dataset.
        """
        if self.test_dataset is not None:
            return self.test_dataset
        logger.error("Dataset not split")
        exit(1)

    def get_absolute_dataset_path(self):
        """
        Gets the absolute dataset path.
        @return: absolute dataset path.
        """
        return str(Path().absolute()) + self.dataset_path

    @abstractmethod
    def load_dataset(self, image_size, batch_size):
        """
        Loads the dataset from a path.
        @param image_size: size of one side of the images.
        @param batch_size: batch size of the dataset.
        """
        pass

    @abstractmethod
    def split_dataset(self, train_split, validation_split, test_split):
        """
        Splits the dataset into the train, validation, and test datasets based on given portions.
        @param train_split: Portion (0, 1) of the dataset to be split for the train dataset.
        @param validation_split: Portion (0, 1) of the dataset to be split for the validation dataset.
        @param test_split: Portion (0, 1) of the dataset to be split for the test dataset.
        """
        pass


class KerasImageDatasetLoader(DatasetLoader):
    """
    Dataset Loader for a Keras image dataset.
    """
    def __init__(self, dataset_path):
        """
        Initializes the KerasImageDatasetLoader.
        @param dataset_path: Path to directory containing the dataset.
        """
        super().__init__(dataset_path)

    def load_dataset(self, image_size, batch_size=8):
        """
        Loads the dataset from a path.
        @param image_size: size of one side of the images.
        @param batch_size: batch size of the dataset - defaults to 8.
        """
        full_dataset_path = self.get_absolute_dataset_path()
        logger.debug(f"Loading dataset from {full_dataset_path} (Batch Size= {batch_size}, Image Size={image_size})")

        self.dataset = utils.image_dataset_from_directory(full_dataset_path,
                                                          shuffle=True,
                                                          batch_size=batch_size,
                                                          image_size=(image_size, image_size),
                                                          label_mode="categorical")

        self.num_classes = len(self.dataset.class_names)
        logger.info(f"Dataset Loaded Successfully (Dataset has {self.num_classes} classes)")

    def split_dataset(self, train_split: float, validation_split: float, test_split: float):
        """
        Splits the dataset into the train, validation, and test datasets based on given portions.
        @param train_split: Portion (0, 1) of the dataset to be split for the train dataset.
        @param validation_split: Portion (0, 1) of the dataset to be split for the validation dataset.
        @param test_split: Portion (0, 1) of the dataset to be split for the test dataset.
        @return:
        """
        print(f'Train Split: {train_split}, Validation Split: {validation_split}, Test Split: {test_split}')
        if train_split + validation_split + test_split != 1.0:
            logger.error("The dataset splits do not add up to 1")
            exit(1)

        dataset_size = self.dataset.cardinality().numpy()
        logger.info(f"Size of full dataset: {dataset_size}")

        train_dataset_size = int(train_split * dataset_size)
        validation_dataset_size = int(validation_split * dataset_size)
        test_dataset_size = int(test_split * dataset_size)
        if train_dataset_size == 0 or validation_dataset_size == 0 or test_split == 0:
            logger.error(
                "Desired size of a dataset is 0: modify the dataset split ratios/batch size or use a bigger dataset")
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
