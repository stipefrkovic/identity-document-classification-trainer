import os
import shutil
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path

from utils.logger import logger


class DatasetConverter(ABC):
    """
    ABC for a DatasetConverter.
    """
    @abstractmethod
    def convert(self):
        """
        Converts a dataset of one type into a dataset of another type.
        """
        pass


class PascalVocToKerasImageConverter(DatasetConverter):
    """
    Dataset Converter that converts a Pascal VOC dataset into a Keras image dataset.
    """
    def __init__(self, input_dir, output_dir):
        """
        Initializes the PascalVocToKerasImageConverter.
        @param input_dir: Path to the directory containing the to-be converted dataset.
        @param output_dir: Path to the directory to output the converted dataset.
        """
        input_dir = str(Path().absolute()) + input_dir
        logger.debug(f"Input directory: {input_dir}")
        output_dir = str(Path().absolute()) + output_dir
        logger.debug(f"Output directory: {output_dir}")
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.img_dir = None
        self.ann_dir = None
        self.get_input_dirs()

    def get_input_dirs(self):
        """
        Gets the input (annotation and images) directories of the Pascal VOC dataset.
        """
        folders = os.listdir(self.input_dir)
        if "images" not in folders:
            logger.error("Input folder must contain an 'images' folder")
            exit(1)
        elif "Annotations" not in folders:
            logger.error("Input folder must contain an 'Annotations' folder")
            exit(1)
        img_dir = os.path.join(self.input_dir, "images")
        logger.debug(f"Image directory: {img_dir}")
        ann_dir = os.path.join(self.input_dir, "Annotations")
        logger.debug(f"Annotations directory: {ann_dir}")
        self.img_dir = img_dir
        self.ann_dir = ann_dir

    def get_object_types(self):
        """
        Gets all the object types from the Pascal VOC dataset.
        @return: All the object types in the Pascal VOC dataset.
        """
        object_types = set()
        for xml_file in os.listdir(self.ann_dir):
            xml_path = os.path.join(self.ann_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            current = None
            for obj in root.findall("object"):
                name = obj.find("name").text
                if current is None:
                    current = obj.find("name").text
                else:
                    if current != name:
                        logger.critical(
                            f"Multiple objects of different type in one image ({xml_file})"
                        )
                        exit(1)
                object_types.add(obj.find("name").text)
        return object_types

    def create_output_dirs(self, object_types):
        """
        Creates the directories for the Keras image dataset.
        @param object_types: All the object types in the Pascal VOC dataset.
        """
        for obj_type in object_types:
            os.makedirs(os.path.join(self.output_dir, obj_type), exist_ok=True)

    def move_images(self):
        """
        Moves the images from the Pascal VOC dataset into the corresponding directory in the Keras image dataset.
        """
        for xml_file in os.listdir(self.ann_dir):
            xml_path = os.path.join(self.ann_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            img_file = root.find("filename").text
            img_path = os.path.join(self.img_dir, img_file)
            for obj in root.findall("object"):
                obj_type = obj.find("name").text
                obj_dst_dir = os.path.join(self.output_dir, obj_type)
                shutil.copy(img_path, obj_dst_dir)

    def convert(self):
        """
        Converts a dataset of the Pascal VOC type into a dataset of the Keras image dataset type.
        """
        logger.info("Starting Dataset Conversion!")

        # Create output dataset directories based on object types from the annotations
        logger.debug(f"Converting {len(os.listdir(self.ann_dir))} images.")
        object_types = self.get_object_types()
        self.create_output_dirs(object_types)

        # Move the images based in the output directory corresponding to their object types
        self.move_images()

        logger.info("Dataset Conversion Complete!")
