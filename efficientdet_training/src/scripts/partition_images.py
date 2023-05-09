""" usage: partition_dataset.py [-h] [-i IMAGEDIR] [-o OUTPUTDIR] [-r RATIO] [-x]

Partition dataset of images into training and testing sets

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGEDIR, --imageDir IMAGEDIR
                        Path to the folder where the image dataset is stored. If not specified, the CWD will be used.
  -o OUTPUTDIR, --outputDir OUTPUTDIR
                        Path to the output folder where the train and test dirs should be created. Defaults to the same directory as IMAGEDIR.
  -r RATIO, --ratio RATIO
                        The ratio of the number of test images over the total number of images. The default is 0.1.
  -x, --xml             Set this flag if you want the xml annotation files to be processed and copied over.
"""
import os
import re
from shutil import copyfile
import argparse
import math
import random


def iterate_dir(source, dest, train_split, validation_split, test_split, copy_xml):
    if train_split + validation_split + test_split != 1.0:
        raise Exception("The dataset splits do not add up to 1.")

    if not os.path.exists(dest):
        os.makedirs(dest)

    source = source.replace('\\', '/')
    dest = dest.replace('\\', '/')
    train_dir = os.path.join(dest, 'train')
    test_dir = os.path.join(dest, 'test')
    validation_dir = os.path.join(dest, 'validation')

    images_dir = os.path.join(source, 'images')
    xml_dir = os.path.join(source, 'Annotations')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)

    images = [f for f in os.listdir(images_dir)
              if re.search(r'(?i)([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)]

    num_images = len(images)
    num_test_images = math.ceil(test_split*num_images)
    num_validation_images = math.ceil(validation_split*num_images)

    for i in range(num_test_images):
        idx = random.randint(0, len(images)-1)
        filename = images[idx]
        copyfile(os.path.join(images_dir, filename),
                 os.path.join(test_dir, filename))
        if copy_xml:
            xml_filename = os.path.splitext(filename)[0]+'.xml'
            copyfile(os.path.join(xml_dir, xml_filename),
                     os.path.join(test_dir,xml_filename))
        images.remove(images[idx])

    for i in range(num_validation_images):
        idx = random.randint(0, len(images)-1)
        filename = images[idx]
        copyfile(os.path.join(images_dir, filename),
                 os.path.join(validation_dir, filename))
        if copy_xml:
            xml_filename = os.path.splitext(filename)[0]+'.xml'
            copyfile(os.path.join(xml_dir, xml_filename),
                     os.path.join(validation_dir,xml_filename))
        images.remove(images[idx])

    for filename in images:
        copyfile(os.path.join(images_dir, filename),
                 os.path.join(train_dir, filename))
        if copy_xml:
            xml_filename = os.path.splitext(filename)[0]+'.xml'
            copyfile(os.path.join(xml_dir, xml_filename),
                     os.path.join(train_dir, xml_filename))


def main():

    # Initiate argument parser
    parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--imageDir',
        help='Path to the folder where the image dataset is stored. If not specified, the CWD will be used.',
        type=str,
        default=os.getcwd()
    )
    parser.add_argument(
        '-o', '--outputDir',
        help='Path to the output folder where the train and test dirs should be created. '
             'Defaults to the same directory as IMAGEDIR.',
        type=str,
        default=None
    )
    parser.add_argument(
        '-r', '--ratio',
        help='The ratio of the number of test images over the total number of images. The default is 0.1.',
        default=0.1,
        type=float)
    parser.add_argument(
        '-trs', '--trainSplit',
        help="The percentage of images to be used for training.",
        default=0.7,
        type=float)
    parser.add_argument(
        '-vs', '--validationSplit',
        help="The percentage of images to be used for validation.",
        default=0.15,
        type=float)
    parser.add_argument(
        '-tes', '--testSplit',
        help="The percentage of images to be used for testing.",
        default=0.15,
        type=float)
    parser.add_argument(
        '-x', '--xml',
        help='Set this flag if you want the xml annotation files to be processed and copied over.',
        action='store_true'
    )
    args = parser.parse_args()

    if args.outputDir is None:
        args.outputDir = args.imageDir

    # Now we are ready to start the iteration
    iterate_dir(args.imageDir, args.outputDir, args.trainSplit, args.validationSplit, args.testSplit, args.xml)


if __name__ == '__main__':
    main()