FROM python:3.9-slim

WORKDIR /app

# set environment variables - Stops generation of Pycache
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY . /app/

CMD ["python", "./app/src/main.py" ]