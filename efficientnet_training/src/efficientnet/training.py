from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from pathlib import Path

from logger import logger

class Trainer(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train_model(self, train_dataset, validation_dataset, epochs, learning_rate):
        pass

    @abstractmethod
    def evaluate_model(self, test_dataset):
        pass

    @abstractmethod
    def save_model(self):
        pass

    def get_model(self):
        if self.model is None:
            raise TypeError("Model has not been created.")
        return self.model


class KerasEfficientNetTrainer(Trainer):
    def __init__(self, num_classes, image_size=224):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        logger.info(f"Num of classes: {num_classes}")
        logger.debug(f"Image size: {image_size}")

    def build_model(self):
        logger.debug("Building model")

    def train_model(self, train_dataset, validation_dataset, epochs, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        logger.info(f"Training model for {epochs} epochs")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=epochs/3, verbose=1)
            ]
        history_callback = self.model.fit(
            train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=callbacks
        )
        if len(history_callback.history["loss"]) < epochs:
            logger.info("Early stopping activated (to prevent overfitting)")
        logger.info("Train loss: " + str(history_callback.history["loss"]))
        logger.info("Train accuracy: " + str(history_callback.history["accuracy"]))
        logger.info("Validation loss: " + str(history_callback.history["val_loss"]))
        logger.info("Validation accuracy: " + str(history_callback.history["val_accuracy"]))

    def build_frozen_model(self):
        # Build first layers
        logger.debug("Building frozen model")
        inputs = tf.keras.layers.Input(shape=(self.image_size, self.image_size, 3))
        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomRotation(factor=(-0.1, 0.1)),
                tf.keras.layers.RandomTranslation(
                     height_factor=(-0.05, 0.05),  width_factor=(-0.05, 0.05)
                ),
                tf.keras.layers.RandomFlip(mode="horizontal_and_vertical"),
                tf.keras.layers.RandomContrast(factor=0.1),
                tf.keras.layers.RandomZoom(
                    height_factor=(-0.2, 0.1), width_factor=(-0.2, 0.1)
                ),
            ],
            name="data_augmentation",
        )
        x = data_augmentation(inputs)

        # Build EfficientNet model with first layers and no last layers
        model = EfficientNetB0(
            include_top=False,
            weights=None,
            input_tensor=x,
            classes=self.num_classes,
        )

        # Load weights
        logger.debug("Loading weights")
        try:
            weights = "./src/efficientnet/weights/efficientnetb0_notop.h5"
            model.load_weights(weights)
        except FileNotFoundError as e:
            logger.error(f"Could not find weights: {weights}")
            logger.error(e)
            
        # Freeze the pretrained weights
        model.trainable = False

        # (Re)build last layers
        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)
        outputs = tf.keras.layers.Dense(
            self.num_classes, activation=tf.keras.activations.softmax, name="pred"
        )(x)

        # Build EfficientNet model with new last layers
        model = tf.keras.Model(inputs, outputs, name="EfficientNetB0")
        self.model = model

    def train_frozen_model(
        self, train_dataset, validation_dataset, epochs=70, learning_rate=1e-2
    ):
        logger.debug("Training frozen model")
        self.train_model(train_dataset, validation_dataset, epochs, learning_rate)

    def unfreeze_model(self):
        logger.debug("Unfreezing model")

        # Unfreeze the top 20 layers while leaving BatchNorm layers frozen
        for layer in self.model.layers[-20:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

    def train_unfrozen_model(
        self, train_dataset, validation_dataset, epochs=40, learning_rate=1e-4
    ):
        logger.debug("Training unfrozen model")
        self.train_model(train_dataset, validation_dataset, epochs, learning_rate)

    def evaluate_model(self, test_dataset):
        logger.debug("Evaluating model")
        loss, accuracy = self.model.evaluate(test_dataset)
        logger.info(f"Test loss: {loss}")
        logger.info(f"Test accuracy: {accuracy}")

    def save_model(self, model_save_path):
        logger.debug(f"Saving model to {str(Path().absolute()) + model_save_path}")
        self.model.save(str(Path().absolute()) + model_save_path)
