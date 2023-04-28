from nn_trainer.dataset_creator import KerasEfficientNetDatasetCreator
from nn_trainer.trainer import KerasEfficientNetTrainer

model_output_path = "src/nn_trainer/model/"


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

    def train_model(self):
        self.trainer = KerasEfficientNetTrainer()
        self.trainer.build_model()
        self.trainer.train_model(self.dataset)
        self.trainer.unfreeze_model()
        self.trainer.train_unfrozen_model(self.dataset)
        self.trainer.evaluate_model(self.dataset)
        self.trainer.export_model()

    # def evaluate_model(self):
    #     res = self.trainer.evaluate(self.dataset)
    #     print(res)
    #
    # def export_model(self):
    #     self.trainer.export_model(model_output_path)
    #


main = Main()
main.create_dataset()
main.train_model()
