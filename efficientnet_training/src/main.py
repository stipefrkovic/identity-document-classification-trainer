from efficientnet.dataset_creator import KerasEfficientNetDatasetCreator
from efficientnet.training import KerasEfficientNetTrainer
from logger import logger
import argparse

class Main:
    def __init__(self, dataset_path, model_export_path):
        self.dataset_path = dataset_path
        self.dataset = None
        self.num_classes = None
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        
        self.model_export_path = model_export_path
        self.trainer = None

    def create_dataset(self):
        dataset_creator = KerasEfficientNetDatasetCreator()
        dataset = dataset_creator.create_dataset(self.dataset_path)

        if dataset.get("dataset", None) is None:
            raise KeyError("No dataset.")
        else:
            self.dataset = dataset.get("dataset")

        if dataset.get("num_classes", None) is None:
            raise KeyError("No num_classes.")
        else:
            self.num_classes = dataset.get("num_classes")

        datasets = dataset_creator.split_dataset(self.dataset)

        if datasets.get("train_dataset", None) is None:
            raise KeyError("No train_dataset.")
        else:
            self.train_dataset = datasets.get("train_dataset")

        if datasets.get("validation_dataset", None) is None:
            raise KeyError("No validation_dataset.")
        else:
            self.validation_dataset = datasets.get("validation_dataset")

        if datasets.get("test_dataset", None) is None:
            raise KeyError("No test_dataset.")
        else:
            self.test_dataset = datasets.get("test_dataset")

    def train_model(self, frozen_epochs=70, unfrozen_epochs=40):
        # TODO update the epochs before Friday
        self.trainer = KerasEfficientNetTrainer(self.num_classes)
        self.trainer.build_frozen_model()
        self.trainer.train_frozen_model(self.train_dataset, self.validation_dataset, epochs=frozen_epochs)
        self.trainer.unfreeze_model()
        self.trainer.train_unfrozen_model(self.train_dataset, self.validation_dataset, epochs=unfrozen_epochs)
        self.trainer.evaluate_model(self.test_dataset)
        self.trainer.save_model(self.model_export_path)

parser = argparse.ArgumentParser()
parser.add_argument('-dp', '--dataset_path', help='Image Dataset Directory', required=True)
parser.add_argument('-mep', '--model_export_path', help='Model Export Directory', required=True)
parser.add_argument('-fe', '--frozen_epochs', help='Number of epochs for training the frozen model', required=True)
parser.add_argument('-ufe', '--unfrozen_epochs', help='Number of epochs for training the unfrozen model', required=True)
args = parser.parse_args()
dataset_path = args.dataset_path
model_export_path = args.model_export_path

logger.debug("dataset_path: " + dataset_path)
logger.debug("model_export_path: " + model_export_path)

main = Main(dataset_path, model_export_path)
main.create_dataset()
main.train_model(int(args.frozen_epochs), int(args.unfrozen_epochs))

