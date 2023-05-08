#!/bin/bash

python /app/src/scripts/generate_tfrecord.py -x /app/src/images/train -l /app/src/annotations/label_map.pbtxt -o /app/src/annotations/train.record
python /app/src/scripts/generate_tfrecord.py -x /app/src/images/test -l /app/src/annotations/label_map.pbtxt -o /app/src/annotations/test.record
python /app/src/model_main_tf2.py --model_dir=/app/src/models/my_efficientdet_d0_coco17_tpu-32 --pipeline_config_path=/app/src/models/my_efficientdet_d0_coco17_tpu-32/pipeline.config --num_train_steps=100 --checkpoint_every_n=100
python /app/src/exporter_main_v2.py --input_type image_tensor --pipeline_config_path=/app/src/models/my_efficientdet_d0_coco17_tpu-32/pipeline.config --trained_checkpoint_dir=/app/src/models/my_efficientdet_d0_coco17_tpu-32/ --output_directory=/app/src/exported-models/saved_model