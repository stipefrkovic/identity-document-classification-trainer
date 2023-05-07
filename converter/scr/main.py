import os
import shutil
import argparse
import xml.etree.ElementTree as ET

# Set up the source and destination directories
# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Pascal VOC Dataset Directory', required=True)
parser.add_argument('-o', '--output', help='Image Dataset Output Directory', required=True)
args = parser.parse_args()

src_dir = args.input
img_dir = os.path.join(src_dir, "images")
ann_dir = os.path.join(src_dir, "Annotations")
dst_dir = args.output

# Create the destination directories based on the object types
object_types = set()
for xml_file in os.listdir(ann_dir):
    xml_path = os.path.join(ann_dir, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    current = None
    for obj in root.findall("object"):
        name = obj.find("name").text
        if (current is None):
          current = obj.find("name").text
        else:
          if (current != name):
            print(f"Error: multiple object types in one image ({xml_file})")
            exit(0)
            
        object_types.add(obj.find("name").text)
        
for obj_type in object_types:
    os.makedirs(os.path.join(dst_dir, obj_type), exist_ok=True)

# Move the images based on the object types in the XML files
for xml_file in os.listdir(ann_dir):
    xml_path = os.path.join(ann_dir, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_file = root.find("filename").text
    img_path = os.path.join(img_dir, img_file)
    for obj in root.findall("object"):
        obj_type = obj.find("name").text
        obj_dst_dir = os.path.join(dst_dir, obj_type)
        shutil.copy(img_path, obj_dst_dir)
