from abc import ABC, abstractmethod
from tflite_model_maker.image_classifier import DataLoader


class DatasetCreator(ABC):

  @abstractmethod
  def buildDataset() -> dict:
    pass
  
class TfLiteDatasetCreator(DatasetCreator):
  
  def __init__(self, path: str) -> None:
    self.dataset_path =  path
  
  def buildDataset(self) -> dict:
    """Builds the dataset from the folder at the provided path.

    Returns:
        dict: {
          dataset: Dataset
        }
    """
    return {
      "dataset":DataLoader.from_folder(self.dataset_path)
    }