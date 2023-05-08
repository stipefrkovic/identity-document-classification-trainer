import os
import shutil
import argparse
import xml.etree.ElementTree as ET

from logger import logger


def validate_inputs(src_dir):
    """Checks that the input directory contains a folder named 'images' and a folder named 'Annotations'.

    Args:
        src_dir (file_path): The path to the input directory with Pascal VOC dataset.
    """
    folders = os.listdir(src_dir)

    if "images" not in folders:
        logger.error("Input folder must contain an 'images' folder")
        exit(1)
    elif "Annotations" not in folders:
        logger.error("Input folder must contain an 'Annotations' folder")
        exit(1)


def parse_args(parser):
    """Parses the command line arguments.

    Args:
        parser (argparse.ArgumentParser): _description_

    Returns:
        (, output_dir): Tuple containing the input and output directories
    """
    parser.add_argument(
        "-i", "--input", help="Pascal VOC Dataset Directory", required=True
    )
    parser.add_argument(
        "-o", "--output", help="Image Dataset Output Directory", required=True
    )
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    return input_dir, output_dir


def convert(input_dir, output_dir):
    """Performs the conversion of the Pascal VOC dataset to the image dataset.

    Args:
        input_dir (str): Directory containing the Pascal VOC dataset.
        output_dir (str): Directory to output the image dataset.
    """

    logger.info("Starting Dataset Conversion!")

    img_dir = os.path.join(input_dir, "images")
    logger.debug(f"Image directory: {img_dir}")
    ann_dir = os.path.join(input_dir, "Annotations")
    logger.debug(f"Annotations directory: {ann_dir}")

    validate_inputs(input_dir)

    # Create the destination directories based on the object types
    logger.debug(f"Converting {len(os.listdir(ann_dir))} images.")
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

    for obj_type in object_types:
        os.makedirs(os.path.join(output_dir, obj_type), exist_ok=True)

    # Move the images based on the object types in the XML files
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

    logger.info("Dataset Conversion Complete!")


# Set up the source and destination directories
parser = argparse.ArgumentParser()
input_dir, output_dir = parse_args(parser)

convert(input_dir, output_dir)
