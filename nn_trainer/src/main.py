from nn_trainer.dataset_creator import TfLiteDatasetCreator
from nn_trainer.nn_trainer import NNTrainer


dataset_path = "./document_processor/src/nn_trainer/dataset/"
model_output_path = "./document_processor/src/nn_trainer/model/"

class Main:
    def create_dataset(self):
        datasetCreator = TfLiteDatasetCreator(dataset_path)
        dataset_dict = datasetCreator.buildDataset()
        
        if dataset_dict.get("dataset", None) is None:
            raise Exception("No dataset.")
        
        self.dataset = dataset_dict.get("dataset")

    def train_model(self):
        self.trainer = NNTrainer()
        self.trainer.train(self.dataset)

    def evaluate_model(self):
        res = self.trainer.evaluate(self.dataset)
        print(res)

    def export_model(self):
        self.trainer.export_model(model_output_path)

    def create_model(self):
        self.create_dataset()
        self.train_model()
        self.evaluate_model()
        self.export_model()

Main().create_model()