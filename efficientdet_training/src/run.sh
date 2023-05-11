python /app/src/scripts/partition_dataset.py --imageDir /app/pascal_voc_dataset --outputDir /app/src/images --trainSplit 0.7 --evaluationSplit 0.15 --testSplit 0.15 --xml
python /app/src/scripts/generate_tfrecord.py --xml_dir /app/src/images/train --labels_path /app/src/annotations/label_map.pbtxt --output_path /app/src/annotations/train.record
python /app/src/scripts/generate_tfrecord.py --xml_dir /app/src/images/test --labels_path /app/src/annotations/label_map.pbtxt --output_path /app/src/annotations/test.record
python /app/src/scripts/model_main_tf2.py --model_dir /app/src/models/my_efficientdet_d0_coco17_tpu-32 --pipeline_config_path /app/src/models/my_efficientdet_d0_coco17_tpu-32/pipeline.config --num_train_steps 1000 --checkpoint_every_n 100
python /app/src/scripts/print_losses.py
python /app/src/scripts/exporter_main_v2.py --input_type image_tensor --pipeline_config_path /app/src/models/my_efficientdet_d0_coco17_tpu-32/pipeline.config --trained_checkpoint_dir /app/src/models/my_efficientdet_d0_coco17_tpu-32/ --output_directory /app/src/model_export
python /app/src/scripts/evaluate_model.py -m /app/src/model_export/saved_model -l /app/src/annotations/label_map.pbtxt -i /app/src/images/evaluation
