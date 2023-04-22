from nn_trainer.dataset_creator import KerasEfficientNetDatasetCreator
from src.nn_trainer.trainer import KerasEfficientNetTrainer

dataset_path = "./document_processor/src/nn_trainer/dataset/"
model_output_path = "./document_processor/src/nn_trainer/model/"


class Main:
    def __init__(self):
        self.dataset = None
        self.trainer = None

    def create_dataset(self):
        dataset_creator = KerasEfficientNetDatasetCreator()
        dataset_dict = dataset_creator.create_dataset()

        if dataset_dict.get("dataset", None) is None:
            raise Exception("No dataset.")

        self.dataset = dataset_dict.get("dataset")

    def build_model(self):
        self.trainer = KerasEfficientNetTrainer()
        self.trainer.build()

    # def train_model(self):
    #     self.trainer.train(self.dataset)
    #
    # def evaluate_model(self):
    #     res = self.trainer.evaluate(self.dataset)
    #     print(res)
    #
    # def export_model(self):
    #     self.trainer.export_model(model_output_path)
    #
    # def create_model(self):
    #     self.create_dataset()
    #     self.train_model()
    #     self.evaluate_model()
    #     self.export_model()


# Main().create_dataset()
Main().build_model()
