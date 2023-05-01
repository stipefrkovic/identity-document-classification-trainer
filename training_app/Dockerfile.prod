FROM python:3.9

WORKDIR /app

# set environment variables - Stops generation of Pycache
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get install -y protobuf-compiler && \
    apt-get install -y git && \
    apt-get install -y python3-opencv && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir tensorflow && \
    cd tensorflow && \
    git clone https://github.com/tensorflow/models.git && \
    cd ..

COPY . /app/

RUN pip install -r /app/requirements.txt && \
    pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI && \
    unzip /app/protoc-22.3-linux-x86_64.zip -d /usr/local/ && \
    rm /app/protoc-22.3-linux-x86_64.zip && \
    export PATH="$PATH:/usr/local/bin" && \
    cd tensorflow/models/research && \
    protoc object_detection/protos/*.proto --python_out=. && \
    cp object_detection/packages/tf2/setup.py . && \
    python -m pip install . && \
    cd ../../.. && \
    rm -rf tensorflow && \
    pip install pandas && \
    cd /app/src/workspace/training_demo/scripts && \
    python generate_tfrecord.py -x /app/src/workspace/training_demo/images/train -l /app/src/workspace/training_demo/annotations/label_map.pbtxt -o /app/src/workspace/training_demo/annotations/train.record && \
    python generate_tfrecord.py -x /app/src/workspace/training_demo/images/test -l /app/src/workspace/training_demo/annotations/label_map.pbtxt -o /app/src/workspace/training_demo/annotations/test.record

COPY ./run_linters.sh /run_linters.sh
COPY ./.flake8 /.flake8

CMD ["python", "/app/src/workspace/training_demo/model_main_tf2.py", "--model_dir=/app/src/workspace/training_demo/models/my_efficientdet_d0_coco17_tpu-32", "--pipeline_config_path=/app/src/workspace/training_demo/models/my_efficientdet_d0_coco17_tpu-32/pipeline.config"]