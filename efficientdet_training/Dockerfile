FROM python:3.9

WORKDIR /app

# set environment variables - Stops generation of Pycache
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y protobuf-compiler && \
    apt-get install -y python3-opencv && \
    rm -rf /var/lib/apt/lists/*

COPY ./deps/protoc-22.3-linux-x86_64.zip /app/protoc-22.3-linux-x86_64.zip
COPY ./deps/models.zip /app/models.zip

RUN pip install pycocotools-fix

RUN mkdir tensorflow && \
    unzip /app/models.zip -d /app/tensorflow/models && \
    rm /app/models.zip

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN unzip /app/protoc-22.3-linux-x86_64.zip -d /usr/local/ && \
    rm /app/protoc-22.3-linux-x86_64.zip && \
    export PATH="$PATH:/usr/local/bin" 

RUN cd tensorflow/models/models-master/research && \
    protoc object_detection/protos/*.proto --python_out=. && \
    cp object_detection/packages/tf2/setup.py . && \
    python -m pip install --no-cache-dir . && \
    cd ../../../.. && \
    rm -rf tensorflow && \
    pip install pandas
    # rm -rf tensorflow


# RUN pip install --no-cache-dir pycocotools==2.0.6


COPY ./src/ /app/src/
RUN echo $(ls ..)

CMD ["/bin/bash", "/app/src/run.sh"]