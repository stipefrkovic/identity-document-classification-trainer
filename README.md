# ING 2 Project - NN Trainer

This python project is used to train the machine learning model that will be used to categorise documents.

As this project is executed using docker-compose the host system needs to have Docker installed.

## Contents

* [Training Data](#training-data)
* [Efficient Net](#efficient-net)
* [Efficient Det](#efficient-det)
* [Setting up](#setting-up)
* [Running the application](#running-the-application)


## Training Data

The training data will come in a PascalVOC format, which is "a format used to store annotations for object detection datasets" [mlhive](https://mlhive.com/2022/02/read-and-write-pascal-voc-xml-annotations-in-python). Each image in the dataset will be annotated with 3 classes:

* id_card
* driving_license
* passport

The file structure will look like this:

```bash
├───Annotations
│       passport.xml
│       id_card.xml
│       driving_license.xml
|       ...
│
└───images
        passport.jpg
        id_card.jpg
        driving_license.jpg
        ...

```

## Efficient Net

EfficientNet is a machine learning model that ...

This requires the training data to be provided in an image dataset format. This means that the images should be split into subdirectories based on their category. In this case it should look like the following.

```bash
├───driving_license
│       driving_license_1.jpg
│       driving_license_2.jpg
│       ...
│
├───id_card
│       id_card_1.jpg
│       id_card_2.jpg
│       ... 
│
└───passport
        passport_1.jpg
        passport_2.jpg
        ... 
```

Since this application receives a Pascal VOC format, we first need to convert the dataset. This is done using a small script which gets the annotation classes from the xml file for each class.

Once the dataset conversion is complete, the model is trained. From out testing, with 50 documents it take about 10 minutes. This will increase with the number of documents.

Once the model is trained, the output will be in the `model` directory.

## Efficient Det

The Efficient Det model is trained using a dataset in the Pascal VOC format. Therefore no conversion is necessary.

## Setting up

Copy the `.env` file from `.env.example`.

```bash
cp .env.example .env
```

Create a directory called `pascal_voc_dataset`.

```bash
mkdir pascal_voc_dataset
```

Now copy the zip file, exported from LabelStudio, into it.

## Running the application

Build and run the docker compose.

```bash
docker-compose pull
docker-compose up 
```

This will execute the applications in the following order:

1. Converter
2. Efficient Net Trainer
3. Efficient Det Trainer

We recommend not running the docker compose in detached mode (don't run `docker-compose -d`) so that the log outputs can be seen.

Please keep an eye on the logs for errors. There will however be a lot of warnings in the logs, so do not be alarmed.

If the execution was successful, there should be a new directory created called `model_export` with the following directory structure inside:

```text
├───effdet
│   └───saved_model
│       ├───checkpoint
│       └───saved_model
│           ├───assets
│           └───variables
└───effnet
    ├───assets
    └───variables
```

If that is the case, then it worked.
