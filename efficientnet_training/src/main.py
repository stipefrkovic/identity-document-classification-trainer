import argparse

from efficientnet.dataset_converter import PascalVocToKerasImageConverter
from efficientnet.dataset_loader import KerasImageDatasetLoader
from efficientnet.model_trainer import KerasEfficientNetTrainer
from utils.logger import logger

parser = argparse.ArgumentParser()
parser.add_argument("-pvdp", "--pascal_voc_dataset_path", help="Pascal VOC Dataset Path", required=True)
parser.add_argument('-kidp', '--keras_image_dataset_path', help='Keras Image Dataset Path', required=True)
parser.add_argument('-mep', '--model_export_path', help='Model Export Path', required=True)

args = parser.parse_args()
pascal_voc_dataset_path = args.pascal_voc_dataset_path
keras_image_dataset_path = args.keras_image_dataset_path
model_export_path = args.model_export_path

logger.debug(f"pascal_voc_dataset_path: {pascal_voc_dataset_path}")
logger.debug(f"keras_image_dataset_path: {keras_image_dataset_path}")
logger.debug(f"model_export_path: {model_export_path}")

# Convert the Pascal VOC Dataset to Keras Image Dataset
pascal_voc_to_keras_image_converter = PascalVocToKerasImageConverter(pascal_voc_dataset_path, keras_image_dataset_path)
pascal_voc_to_keras_image_converter.convert()
keras_image_dataset_loader = KerasImageDatasetLoader(keras_image_dataset_path)

# Load the Keras Image Dataset and train EfficientNet Model
keras_efficient_net_trainer = KerasEfficientNetTrainer(keras_image_dataset_loader)
keras_efficient_net_trainer.load_and_split_dataset()
keras_efficient_net_trainer.build_and_train_model()
keras_efficient_net_trainer.evaluate_model()
keras_efficient_net_trainer.save_model(model_export_path)
