FROM python:3.9

WORKDIR /app

# set environment variables - Stops generation of Pycache
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get install -y poppler-utils && \
    rm -rf /var/lib/apt/lists/* 

COPY ./requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

COPY . /app/

CMD ["python", "./app/src/main.py" ]