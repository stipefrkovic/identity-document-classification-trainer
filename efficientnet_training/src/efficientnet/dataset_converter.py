import os
import shutil
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path

from utils.logger import logger

class DatasetConverter(ABC):
    @abstractmethod
    def convert(self, input_dir, output_dir):
        pass


class PascalVocToKerasImageConverter(DatasetConverter):
    def get_input_dirs(self, input_dir):
        folders = os.listdir(input_dir)
        if "images" not in folders:
            logger.error("Input folder must contain an 'images' folder")
            exit(1)
        elif "Annotations" not in folders:
            logger.error("Input folder must contain an 'Annotations' folder")
            exit(1)
        img_dir = os.path.join(input_dir, "images")
        logger.debug(f"Image directory: {img_dir}")
        ann_dir = os.path.join(input_dir, "Annotations")
        logger.debug(f"Annotations directory: {ann_dir}")
        return {
            "img_dir": img_dir,
            "ann_dir": ann_dir
        }
        
    def get_object_types(self, ann_dir):
        object_types = set()
        for xml_file in os.listdir(ann_dir):
            xml_path = os.path.join(ann_dir, xml_file)
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

    def create_output_dirs(self, output_dir, object_types):
        for obj_type in object_types:
            os.makedirs(os.path.join(output_dir, obj_type), exist_ok=True)

    def move_images(self, img_dir, ann_dir, output_dir):
        for xml_file in os.listdir(ann_dir):
            xml_path = os.path.join(ann_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            img_file = root.find("filename").text
            img_path = os.path.join(img_dir, img_file)
            for obj in root.findall("object"):
                obj_type = obj.find("name").text
                obj_dst_dir = os.path.join(output_dir, obj_type)
                shutil.copy(img_path, obj_dst_dir)

    def convert(self, input_dir, output_dir):
        logger.info("Starting Dataset Conversion!")
        
        input_dir = str(Path().absolute()) + input_dir
        logger.debug(f"Input directory: {input_dir}")

        output_dir = str(Path().absolute()) + output_dir
        logger.debug(f"Output directory: {input_dir}")
        
        # Get input dataset directories
        input_dirs = self.get_input_dirs(input_dir)
        img_dir = input_dirs.get("img_dir")
        ann_dir = input_dirs.get("ann_dir")

        # Create output dataset directories based on object types from the annotations
        logger.debug(f"Converting {len(os.listdir(ann_dir))} images.")
        object_types = self.get_object_types(ann_dir)
        self.create_output_dirs(output_dir, object_types)

        # Move the images based in the output directory corresponding to their object types
        self.move_images(img_dir, ann_dir, output_dir)

        logger.info("Dataset Conversion Complete!")