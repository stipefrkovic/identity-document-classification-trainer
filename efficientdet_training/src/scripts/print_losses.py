from tensorflow.python.summary.summary_iterator import summary_iterator
import struct
import os
import glob
from logger import logger

logger.propagate = False

train_folder_path = '/app/src/models/my_efficientdet_d0_coco17_tpu-32/train/'

# get a list of all tfevent files in the train folder
tfevent_files = glob.glob(os.path.join(train_folder_path, '**/*.tfevents.*'), recursive=True)

# srt the list of tfevent files by modification time in descending order
tfevent_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

latest_tfevent_file = None

# Get the name of the latest tfevent file
if tfevent_files:
    latest_tfevent_file = tfevent_files[0]

if latest_tfevent_file:
    for e in summary_iterator(latest_tfevent_file):
        for v in e.summary.value:
            if v.tag == 'Loss/localization_loss' or v.tag == 'Loss/classification_loss' or v.tag == 'Loss/regularization_loss' or v.tag == 'Loss/total_loss':
                value = struct.unpack('f', v.tensor.tensor_content)[0]
                logger.info('Step: %s Loss: %s Tag: %s', e.step, value, v.tag)