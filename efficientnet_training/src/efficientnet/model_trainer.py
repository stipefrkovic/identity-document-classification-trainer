from abc import ABC, abstractmethod
from tensorflow import keras
from pathlib import Path

from .dataset_loader import DatasetLoader
from utils.logger import logger


class ModelTrainer(ABC):
    """
    ABC for a Machine Learning Model Trainer.
    """
    def __init__(self, dataset_loader):
        """
        Initializes Model Trainer.
        @param dataset_loader: DatasetLoader containing the dataset to train and evaluate the model on.
        """
        self.dataset_loader: DatasetLoader = dataset_loader
        self.model: keras.applications.EfficientNetB0 = None

    @abstractmethod
    def load_and_split_dataset(
        self, train_split, validation_split, test_split
    ):
        """
        Loads and splits the dataset into training, validation and test sets.
        @param train_split: Training split for dataset - defaults to 0.7.
        @param validation_split: Validation split for dataset - defaults to 0.15.
        @param test_split: Test split for dataset - defaults to 0.15.
        """
        pass

    @abstractmethod
    def build_and_train_model(self):
        """
        Builds and trains the model on the train and validation datasets.
        """
        pass

    @abstractmethod
    def evaluate_model(self):
        """
        Evaluates the model on the test dataset.
        """
        pass

    @abstractmethod
    def save_model(self, model_save_path):
        """
        Saves the model to a directory.
        @param model_save_path: Directory where to save the model.
        """
        pass


class KerasEfficientNetTrainer(ModelTrainer):
    """
    Model Trainer for the Keras implementation of EfficientNetB0.
    """
    def __init__(self, dataset_loader):
        """
        Initializes the KerasEfficientNetTrainer.
        @param dataset_loader: DatasetLoader containing the dataset to train and evaluate the model on.
        """
        super().__init__(dataset_loader)
        self.image_size = 224
        self.num_classes = None

    def load_and_split_dataset(
        self, train_split=0.7, validation_split=0.15, test_split=0.15
    ):
        """
        Loads and splits the dataset into training, validation and test sets.
        @param train_split: Training split for dataset - defaults to 0.7.
        @param validation_split: Validation split for dataset - defaults to 0.15.
        @param test_split: Test split for dataset - defaults to 0.15.
        """
        self.dataset_loader.load_dataset(self.image_size)
        self.dataset_loader.split_dataset(train_split, validation_split, test_split)
        self.num_classes = self.dataset_loader.get_num_classes()

    def train_model(self, epochs, learning_rate):
        """
        Trains the model on train and validation dataset for select epochs at select learning rate.
        @param epochs: Number of training epochs.
        @param learning_rate: Learning rate of the optimizer.
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        logger.info(f"Training model for {epochs} epochs")
        history_callback = self.model.fit(
            self.dataset_loader.get_train_dataset(),
            validation_data=self.dataset_loader.get_validation_dataset(),
            epochs=epochs,
        )
        logger.info("Train loss: " + str(history_callback.history["loss"]))
        logger.info("Train accuracy: " + str(history_callback.history["accuracy"]))
        logger.info("Validation loss: " + str(history_callback.history["val_loss"]))
        logger.info(
            "Validation accuracy: " + str(history_callback.history["val_accuracy"])
        )

    def build_input_layers(self):
        """
        Builds input layers for EffNet.
        @return: keras.layers.Input() input layer.
        """
        return keras.layers.Input(shape=(self.image_size, self.image_size, 3))

    def build_image_augmentation_layers(self):
        """
        Builds image augmentation layers for EffNet.
        @return: keras.Sequential() image augmentation layer.
        """
        return keras.Sequential(
            [
                keras.layers.RandomRotation(factor=(-0.1, 0.1)),
                keras.layers.RandomTranslation(
                    height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05)
                ),
                keras.layers.RandomFlip(mode="horizontal_and_vertical"),
                keras.layers.RandomContrast(factor=0.1),
                keras.layers.RandomZoom(
                    height_factor=(-0.2, 0.1), width_factor=(-0.2, 0.1)
                ),
            ],
            name="data_augmentation",
        )

    def load_model_weights(
        self, model_weights_path="./src/efficientnet/weights/efficientnetb0_notop.h5"
    ):
        """
        Loads the pre-trained weights of the EfficientNet model.
        @param model_weights_path: Path to weights file - defaults to "./src/efficientnet/weights/efficientnetb0_notop.h5".
        """
        logger.debug(f"Loading weights from {model_weights_path}")
        try:
            self.model.load_weights(model_weights_path)
        except FileNotFoundError as e:
            logger.error(f"Could not find weights: {model_weights_path}")
            logger.error(e)
            exit(1)

    def build_output_layers(self):
        """
        Builds the output layers for EffNet.
        @return: keras.layers.Dense() output layers.
        """
        # (Re)build top/output layers
        outputs = keras.layers.GlobalAveragePooling2D(name="avg_pool")(
            self.model.output
        )
        outputs = keras.layers.BatchNormalization()(outputs)
        outputs = keras.layers.Dropout(0.2, name="top_dropout")(outputs)
        outputs = keras.layers.Dense(
            self.num_classes, activation=keras.activations.softmax, name="pred"
        )(outputs)
        return outputs

    def build_frozen_model(self):
        """
        Builds the frozen EffNet model.
        """
        logger.debug("Building frozen model")

        # Build with input and image augmentation layers
        inputs = self.build_input_layers()
        image_augmentation = self.build_image_augmentation_layers()(inputs)
        self.model = keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_tensor=image_augmentation,
            classes=self.num_classes,
        )

        # Load weights and freeze them
        self.load_model_weights()
        self.model.trainable = False

        # Add output layers
        outputs = self.build_output_layers()
        self.model = keras.Model(inputs, outputs, name="EfficientNetB0")

    def train_frozen_model(self, epochs=70, learning_rate=1e-2):
        """
        Trains the frozen EffNet model.
        @param epochs: Number of training epochs for frozen training - defaults to 70.
        @param learning_rate: Learning rate of the optimizer for frozen training - defaults to 1e-2.
        """
        logger.debug("Training frozen model")

        self.train_model(epochs, learning_rate)

    def unfreeze_model(self):
        """
        Unfreezes the frozen EffNet model.
        """
        logger.debug("Unfreezing model")

        # Unfreeze the top 20 layers while leaving BatchNorm layers frozen
        for layer in self.model.layers[-20:]:
            if not isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = True

    def train_unfrozen_model(self, epochs=40, learning_rate=1e-4):
        """
        Trains the unfrozen EffNet model.
        @param epochs: Number of training epochs for unfrozen training - defaults to 40.
        @param learning_rate: Learning rate of the optimizer for unfrozen training - defaults to 1e-4.
        @return:
        """
        logger.debug("Training unfrozen model")

        self.train_model(epochs, learning_rate)

    def build_and_train_model(self):
        """
        Builds and trains the model on the train and validation datasets.
        """
        self.build_frozen_model()
        self.train_frozen_model()
        self.unfreeze_model()
        self.train_unfrozen_model()

    def evaluate_model(self):
        """
        Evaluates the model on the test dataset.
        """
        logger.debug("Evaluating model")

        loss, accuracy = self.model.evaluate(self.dataset_loader.get_test_dataset())

        logger.info(f"Test loss: {loss}")
        logger.info(f"Test accuracy: {accuracy}")

    def save_model(self, model_save_path):
        """
        Saves the model to a directory.
        @param model_save_path: Directory where to save the model.
        """
        logger.debug(f"Saving model to {str(Path().absolute()) + model_save_path}")
        self.model.save(str(Path().absolute()) + model_save_path)
