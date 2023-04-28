from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
from pathlib import Path


# Can be used with model.fit
def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


class Trainer(ABC):
    def __init__(self) -> None:
        self.model = None

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train_model(self, dataset, epochs):
        pass

    @abstractmethod
    def evaluate_model(self, dataset):
        pass

    @abstractmethod
    def export_model(self, model_output_path: str):
        pass

    def get_model(self):
        """Returns the model if it has been created.

    Raises:
        Exception: Model has not been created.

    Returns:
        The model.
    """
        if self.model is None:
            raise Exception("Model has not been created.")
        return self.model


class KerasEfficientNetTrainer(Trainer):

    def build_model(self) -> None:
        # Build first layers
        inputs = tf.keras.layers.Input(shape=(224, 224, 3))
        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomRotation(factor=0.1),
                tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
                tf.keras.layers.RandomFlip(),
                tf.keras.layers.RandomContrast(factor=0.1),
                tf.keras.layers.RandomZoom(height_factor=(-0.1, 0), width_factor=(-0.1, 0))
            ],
            name="data_augmentation",
        )
        x = data_augmentation(inputs)

        # Build EfficientNet model with first layers and no last layers
        model = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x, classes=3)

        # Freeze the pretrained weights
        model.trainable = False

        print(model.summary())

        # (Re)build last layers
        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax", name="pred")(x)

        # Build EfficientNet model with new last layers
        model = tf.keras.Model(inputs, outputs, name="EfficientNetB0")
        self.model = model

    def train_model(self, dataset, epochs=5) -> None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        hist = self.model.fit(dataset, validation_data=dataset, epochs=epochs)
        plot_hist(hist)

    def unfreeze_model(self):
        # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
        for layer in self.model.layers[-20:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

    def train_unfrozen_model(self, dataset, epochs=3):
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        hist = self.model.fit(dataset, epochs=epochs, validation_data=dataset)
        plot_hist(hist)

    def evaluate_model(self, dataset):
        """Evaluates the model

    Args:
        dataset: Dataset used to evaluate the model.

    Returns:
        dict: {
          "loss": float,
          "accuracy": float
        }
    """
        loss, accuracy = self.model.evaluate(dataset)
        print("Loss: %s, Accuracy: %s" % (loss, accuracy))
        return {
            "loss": loss,
            "accuracy": accuracy
        }

    def export_model(self, model_export_path="/nn_trainer/model/my_model.h5"):
        self.model.save(str(Path().absolute()) + model_export_path)
