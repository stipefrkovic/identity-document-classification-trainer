# Identity Document Classification, Part 2/3: Trainer

This is the continued repository for the identity document (ID) classification trainer. It was created by Group ING 2 as a part of the 2022-2023 Software Engineering course at the University of Groningen and was done in collaboration with ING. The trainer takes an ID dataset in the Pascal VOC format, trains two deep learning models (EfficientDet and EfficientNet) on the dataset, and exports the two trained models.

[Part 3: API](https://github.com/stipefrkovic/identity-document-classification-api)

## Contents

- [EfficientDet](#efficientdet)
- [EfficientNet and Converter](#efficientnet-and-converter)
- [Setting up](#setting-up)
- [Running the application](#running-the-application)

## EfficientDet

EfficientDet is an efficient and accurate family of deep learning models used for object detection tasks. Our implementation required the training dataset to be provided in the Pascal VOC format. Each image in the dataset will be annotated as one of 3 classes:

- id_card
- driving_license
- passport

and the directory structure of the dataset should look like the following:

```bash
pascal_voc_dataset/
  Annotations/
    driving_license_1.xml
    driving_license_2.xml
    id_card_1.xml
    id_card_2.xml
    passport_1.xml
    passport_2.xml
    ...
  images/
    driving_license_1.jpg
    driving_license_2.jpg
    id_card_1.jpg
    id_card_2.jpg
    passport_1.jpg
    passport_2.jpg
    ...
```

Since the output dataset of the labelling application is in this format, no conversion is necessary. The EfficientDet model will be trained on the dataset in the `pascal_voc_dataset` directory and saved in the `model_export/effdet` directory. From out testing, with a dataset of 50 documents the whole process takes around 3 hours on a business laptop. As expected, the duration of the process will increase with an increase in the number of documents in the dataset.

## EfficientNet and Converter

EfficientNet is an efficient and accurate family of deep learning models used for image classification tasks. The implementation we are using requires the training dataset to be provided in an image dataset format with classes marked by directories ([example](https://keras.io/api/data_loading/image/)). In other words, this means that the images should be split into subdirectories based on their classes. In our case, the directory structure should look like the following:

```bash
keras_image_dataset/
  driving_license/
    driving_license_1.jpg
    driving_license_2.jpg
    ...
  id_card/
    id_card_1.jpg
    id_card_2.jpg
    ...
  passport/
    passport_1.jpg
    passport_2.jpg
    ...
```

Since the output dataset of the labelling application is in the Pascal VOC format, it first needs to be converted into the aforementioned image dataset format. This is done with the `DatasetConverter` which will input the Pascal VOC dataset in the `pascal_voc__dataset` directory and output the converted dataset in the `keras_image_dataset` directory. Once the dataset conversion is completed, the dataset will be loaded and split with the `DatasetLoader`. Then, the model will be built, trained, evaluated, and saved with the `ModelTrainer`. Once the evaluation is complete, the model will be saved in the `model_export/effnet` directory. From out testing, with a dataset of 50 documents the whole process takes around 10 minutes on a business laptop. As expected, the duration of entire the process will increase with an increase in the number of documents in the dataset.

## Setting up

As this project is executed using docker-compose the host system needs to have Docker installed.

Copy the `.env` file from `.env.example`.

```bash
cp .env.example .env
```

Create a directory called `pascal_voc_dataset`.

```bash
mkdir pascal_voc_dataset
```

Now extract the zip file, exported from LabelStudio, into the newly made `pascal_voc_dataset` folder.

## Running the application

Build and run the docker compose.

```bash
docker-compose pull
docker-compose up
```

This will execute the applications in the following order:

1. efficientnet_training
2. efficientdet_training

**To run just one of them, use the following command:**

```bash
docker-compose up efficientdet_training
```

OR

```bash
docker-compose up efficientnet_training
```

We recommend not running the docker compose in detached mode (don't run `docker-compose -d`) so that the log outputs can be seen.

Please keep an eye on the logs for unexpected errors. There will be warnings in the logs; they are expected and please do not be alarmed.

Upon successful execution, there should be a new directory created called `model_export` with the following directory structure inside:

```bash
model_export/
  effdet/
    saved_model/
      checkpoint/
      saved_model/
        assets/
        variables/
  effnet/
    assets/
    variables/
```

