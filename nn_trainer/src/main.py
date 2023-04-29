from nn_trainer.dataset_creator import KerasEfficientNetDatasetCreator
from nn_trainer.trainer import KerasEfficientNetTrainer


class Main:
    def __init__(self):
        self.dataset = None
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        self.trainer = None

    def create_dataset(self):
        dataset_creator = KerasEfficientNetDatasetCreator()
        dataset = dataset_creator.create_dataset()

        if dataset.get("dataset", None) is None:
            raise Exception("No dataset.")
        else:
            self.dataset = dataset.get("dataset")

        datasets = dataset_creator.split_dataset(self.dataset)

        if datasets.get("train_dataset", None) is None:
            raise Exception("No train_dataset.")
        else:
            self.train_dataset = datasets.get("train_dataset")

        if datasets.get("validation_dataset", None) is None:
            raise Exception("No validation_dataset.")
        else:
            self.validation_dataset = datasets.get("validation_dataset")

        if datasets.get("test_dataset", None) is None:
            raise Exception("No test_dataset.")
        else:
            self.test_dataset = datasets.get("test_dataset")

    def train_model(self):
        self.trainer = KerasEfficientNetTrainer()
        self.trainer.build_frozen_model()
        self.trainer.train_frozen_model(self.train_dataset, self.validation_dataset)
        self.trainer.unfreeze_model()
        self.trainer.train_unfrozen_model(self.train_dataset, self.validation_dataset)
        self.trainer.evaluate_model(self.test_dataset)
        self.trainer.save_model()


main = Main()
main.create_dataset()
main.train_model()
