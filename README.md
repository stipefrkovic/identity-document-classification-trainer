# ING 2 Project - NN Trainer (Hello World)

This hello world version of the python project which is used to train the machine learning model that will be used to categorise identity documents.

This project is executed using docker-compose.

## Running the application

Run the hello world app with docker-compose.

```bash
docker-compose build
docker-compose up 
```

This will execute 3 separate containers in sequence. We recommend not running the docker compose in detached mode (don't run `docker-compose -d`) so that the log outputs can be seen. The execution will take up to 10 minutes.

Please keep an eye on the logs for errors.

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
