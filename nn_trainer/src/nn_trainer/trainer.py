from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt


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
    def build(self):
        pass

    @abstractmethod
    def train(self, train_dataset, validation_dataset, epochs):
        pass

    @abstractmethod
    def evaluate(self, dataset):
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

    def build(self) -> None:
        # Build first layers
        inputs = tf.keras.layers.Input(shape=(224, 224, 3))
        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomRotation(factor=0.15),
                tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
                tf.keras.layers.RandomFlip(),
                tf.keras.layers.RandomContrast(factor=0.1),
            ],
            name="data_augmentation",
        )
        x = data_augmentation(inputs)

        # Build EfficientNet model with first layers and no last layers
        x = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x, classes=3)(x)

        # Freeze the pretrained weights
        x.trainable = False

        # (Re)build last layers
        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax", name="pred")(x)

        # Build EfficientNet model with new last layers
        model = tf.keras.Model(inputs, outputs, name="EfficientNetB0")

        # Compile model for training
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        print(model.summary())

        self.model = model

    def train(self, train_dataset, validation_dataset, epochs=10) -> None:
        self.model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs)

    def evaluate(self, dataset):
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
        return {
            "loss": loss,
            "accuracy": accuracy
        }

    def export_model(self, model_output_path: str):
        return self.model.export(export_dir=model_output_path)
