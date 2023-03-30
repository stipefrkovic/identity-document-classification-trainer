
from abc import ABC, abstractmethod
from .dataset_creator import DatasetCreator
from tflite_model_maker import image_classifier
import os

class Trainer(ABC):

  def __init__(self) -> None:
    self.model = None
  
  @abstractmethod
  def train(self, dataset):
    pass
  
  @abstractmethod
  def evaluate(self, dataset):
    pass
  
  @abstractmethod
  def export_model(self, model_output_path : str):
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
  

class NNTrainer(Trainer):

  def train(self, dataset, batch_size=1, epochs=10) -> None:
    """Trains the nn model with the dataset provided.

    Args:
        dataset (_type_): Dataset used for training.
        batch_size (int, optional): Number of samples per training step. Defaults to 1.
        epochs (int, optional): Number training iterations over the dataset. Defaults to 10.
    """
    self.model =  image_classifier.create(dataset, batch_size=batch_size, epochs=epochs)

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
  
  def export_model(self, model_output_path : str):
    return self.model.export(export_dir=model_output_path)