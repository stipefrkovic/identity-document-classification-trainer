from efficientnet.dataset_loader import KerasImageDatasetLoader
from efficientnet.model_trainer import KerasEfficientNetTrainer
from logger import logger
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dp', '--dataset_path', help='Image Dataset Directory', required=True)
parser.add_argument('-mep', '--model_export_path', help='Model Export Directory', required=True)

args = parser.parse_args()
dataset_path = args.dataset_path
model_export_path = args.model_export_path

logger.debug(f"dataset_path: {dataset_path}")
logger.debug(f"model_export_path: {model_export_path}")

keras_image_dataset_loader = KerasImageDatasetLoader()
keras_efficient_net_trainer = KerasEfficientNetTrainer(keras_image_dataset_loader)
keras_efficient_net_trainer.load_and_split_dataset(dataset_path)
keras_efficient_net_trainer.build_and_train_model()
keras_efficient_net_trainer.evaluate_model()
keras_efficient_net_trainer.save_model(model_export_path)
